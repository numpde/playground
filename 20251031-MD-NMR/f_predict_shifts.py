# /home/ra/repos/playground/20251031-MD-NMR/f_predict_shifts.py
# Timestamp: 2025-11-01 18:20 Africa/Nairobi
#
# Step f: predict NMR shifts (1H/13C) from clustered conformers.
# Outputs per tag:
#   f_predict_shifts/<tag>_cluster_<cid>_shifts.tsv
#   f_predict_shifts/<tag>_fastavg_shifts.tsv
#   f_predict_shifts/<tag>_params.txt
#
# Features
# - Representatives: e_cluster/<tag>_cluster_<cid>_rep.pdb
# - Fast-exchange average; optional Boltzmann reweighting by temperature.
# - Shieldings from PySCF NMR (robust to API differences across versions).
# - TMS reference at same electronic level.
# - Optional pandas TSV loader; falls back to csv.
# - Suppresses PySCF “under testing / not fully tested” warnings.
# - Logging with GPU detection (CuPy / gpu4pyscf / PyTorch).

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("nmrshifts")

# Optional pandas
try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# ── Silence PySCF "prop under testing" warnings (keep real errors visible) ──
warnings.filterwarnings("ignore", message=r"Module .* is under testing", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Module .* is not fully tested", category=UserWarning)

# PySCF
from pyscf import gto, dft
import pyscf

# Optional geometry optimization backends
try:
    from pyscf.geomopt.geomeTRIC import optimize as geom_optimize

    GEOMOPT_AVAILABLE = True
except Exception:
    GEOMOPT_AVAILABLE = False

try:
    from pyscf.geomopt.berny import optimize as berny_optimize

    BERNY_AVAILABLE = True
except Exception:
    BERNY_AVAILABLE = False

# ── Tunables (CLI-overridable) ────────────────────────────────────────────────
DFT_XC_DEFAULT = "b3lyp"
BASIS_DEFAULT = "def2-tzvp"
SCF_MAXCYC = 200
SCF_CONV_TOL = 1e-9
GRAD_CONV_TOL = 3e-4

OUT_DIR = Path("f_predict_shifts")
CLUSTERS_DIR = Path("e_cluster")

PCM_EPS_DEFAULT = 46.7  # DMSO ~298 K
TEMPERATURE_K_DEFAULT = 298.15


# ── Logging setup & GPU detection ────────────────────────────────────────────
def _setup_logging(level: str, quiet: bool) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    if quiet:
        numeric = max(numeric, logging.WARNING)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noisy libs
    logging.getLogger("pyscf").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)


def _detect_gpu() -> Dict[str, object]:
    """Detect GPU backends and devices (CuPy / gpu4pyscf / PyTorch)."""
    info: Dict[str, object] = {
        "gpu_available": False,
        "backend": [],
        "device_count": 0,
        "devices": [],
        "env": {},
    }
    # Capture relevant env toggles for transparency
    for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if k in os.environ:
            info["env"][k] = os.environ[k]

    # CuPy
    try:
        import cupy as cp  # type: ignore
        try:
            n = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            n = 0
        if n > 0:
            info["gpu_available"] = True
            info["backend"].append("cupy")
            info["device_count"] = max(info["device_count"], n)
            for i in range(n):
                try:
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    name = props["name"]
                    if isinstance(name, bytes):
                        name = name.decode()
                    info["devices"].append(str(name))
                except Exception:
                    info["devices"].append(f"device_{i}")
    except Exception:
        pass

    # gpu4pyscf presence
    try:
        import gpu4pyscf  # type: ignore
        info["gpu_available"] = True
        info["backend"].append("gpu4pyscf")
    except Exception:
        pass

    # PyTorch (not used by PySCF, but helpful for cluster introspection)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            info["gpu_available"] = True
            if "torch" not in info["backend"]:
                info["backend"].append("torch")
            n = torch.cuda.device_count()
            info["device_count"] = max(info["device_count"], n)
            for i in range(n):
                info["devices"].append(torch.cuda.get_device_name(i))
    except Exception:
        pass

    # Deduplicate device names
    info["devices"] = list(dict.fromkeys(info["devices"]))
    return info


# ── Utils ────────────────────────────────────────────────────────────────────
def _mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fmt(x: float) -> str:
    return f"{x:.6f}" if np.isfinite(x) else "nan"


# ── Chemistry helpers ────────────────────────────────────────────────────────
def _make_mol(symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int) -> gto.Mole:
    mol = gto.Mole()
    mol.build(
        atom=[(s, tuple(r)) for s, r in zip(symbols, coords_A)],
        unit="Angstrom",
        basis=BASIS_DEFAULT,
        charge=charge,
        spin=spin,
        verbose=0,
    )
    return mol


def _build_rks_or_uks(mol: gto.Mole, xc: str):
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = xc
    mf.max_cycle = SCF_MAXCYC
    mf.conv_tol = SCF_CONV_TOL
    return mf


def _attach_pcm_and_build_scf(mol: gto.Mole, xc: str, solvent_eps: Optional[float], use_pcm: bool):
    mf = _build_rks_or_uks(mol, xc)
    if use_pcm and (solvent_eps is not None):
        try:
            from pyscf import solvent as pyscf_solvent
            mf = pyscf_solvent.ddCOSMO(mf)
            mf.with_solvent.eps = float(solvent_eps)
            LOG.debug("Attached ddCOSMO PCM (eps=%.3f).", solvent_eps)
        except Exception as e:
            LOG.warning("PCM requested but failed to attach: %s; proceeding gas-phase.", e)
    return mf


def _optimize_geometry_pcm(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        solvent_eps: Optional[float],
        backend: str = "auto",
        quiet: bool = False,
) -> np.ndarray:
    # choose backend
    if backend == "geom":
        use_geom = GEOMOPT_AVAILABLE
        use_berny = False
    elif backend == "berny":
        use_geom = False
        use_berny = BERNY_AVAILABLE
    elif backend == "none":
        use_geom = use_berny = False
    else:  # auto
        use_geom = GEOMOPT_AVAILABLE
        use_berny = (not use_geom) and BERNY_AVAILABLE

    if not (use_geom or use_berny):
        if not quiet:
            LOG.info("Geometry optimization skipped (no geomeTRIC/Berny or '--opt none').")
        return coords_A

    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)
    mf_pcm = _attach_pcm_and_build_scf(mol, xc=xc, solvent_eps=solvent_eps, use_pcm=True)

    try:
        if use_geom:
            LOG.info("Geometry optimization via geomeTRIC.")
            _e_opt, mol_opt = geom_optimize(mf_pcm, tol=GRAD_CONV_TOL, maxsteps=200, callback=None)
        else:
            LOG.info("Geometry optimization via Berny.")
            mol_opt = berny_optimize(mf_pcm, maxsteps=200)
        coords_bohr = mol_opt.atom_coords(unit="Bohr")
        return coords_bohr * 0.529177210903
    except Exception as e:
        LOG.warning("Geometry optimization failed: %s; using input geometry.", e)
        return coords_A


# ── PySCF NMR (version-robust) ───────────────────────────────────────────────
def _nmr_tensors_from_mf(mf) -> np.ndarray:
    """
    Return per-atom 3x3 shielding tensors from a PySCF mean-field object.
    Tries several API locations to support multiple PySCF versions:
      - from pyscf.prop.nmr import NMR
      - from pyscf.prop.nmr import rhf/uhf/rks/uks → NMR
      - method on mf: mf.NMR()
    """
    # 1) Unified NMR class
    try:
        from pyscf.prop.nmr import NMR  # type: ignore
        return NMR(mf).kernel()
    except Exception:
        pass

    # 2) Method on mf
    try:
        if hasattr(mf, "NMR"):
            return mf.NMR().kernel()
    except Exception:
        pass

    # 3) Flavor-specific modules
    try:
        from pyscf.prop import nmr as nmrmod  # type: ignore
        if isinstance(mf, dft.rks.RKS) and hasattr(nmrmod, "rks"):
            return nmrmod.rks.NMR(mf).kernel()
        if isinstance(mf, dft.uks.UKS) and hasattr(nmrmod, "uks"):
            return nmrmod.uks.NMR(mf).kernel()
        if hasattr(nmrmod, "rhf") and mf.mol.spin == 0:
            return nmrmod.rhf.NMR(mf).kernel()
        if hasattr(nmrmod, "uhf") and mf.mol.spin != 0:
            return nmrmod.uhf.NMR(mf).kernel()
    except Exception:
        pass

    raise RuntimeError(
        "PySCF NMR interface not found in this environment. "
        "Tried: pyscf.prop.nmr.NMR, nmr.rks/uks/rhf/uhf.NMR, and mf.NMR(). "
        "Please ensure PySCF's property modules are installed/enabled."
    )


def _compute_sigma_iso(symbols, coords_A, charge, spin, xc):
    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)
    mf = _build_rks_or_uks(mol, xc=xc)
    _ = mf.kernel()
    tensors = _nmr_tensors_from_mf(mf)

    # Accept multiple return shapes across PySCF versions:
    # - (natm, 3, 3): full shielding tensors
    # - (natm,): isotropic shieldings already
    # - (natm, 3): principal values → take mean
    arr = np.asarray(tensors)
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[-2] == 3:
        return (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.mean(axis=1).astype(float)

    # Last-resort: try diagonal mean if it looks square-ish
    try:
        diag = np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1)
        return diag.astype(float)
    except Exception:
        raise RuntimeError(f"Unrecognized NMR output shape: {arr.shape}")


def _sigma_to_delta_ppm(symbols: Sequence[str], sigma_iso: np.ndarray, ref_sigma: Dict[str, float]) -> np.ndarray:
    out = np.full_like(sigma_iso, np.nan, dtype=float)
    for i, el in enumerate(symbols):
        elU = el.upper()
        if elU in ("H", "C"):
            out[i] = ref_sigma[elU] - sigma_iso[i]
    return out


def _tms_geometry() -> Tuple[List[str], np.ndarray]:
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []

    def add(a: str, x: float, y: float, z: float) -> None:
        symbols.append(a)
        coords.append((x, y, z))

    add("Si", 0.0, 0.0, 0.0)
    R_SiC, R_CH = 1.86, 1.09
    dirs = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
    for dx, dy, dz in dirs:
        n = (dx * dx + dy * dy + dz * dz) ** 0.5
        ux, uy, uz = dx / n, dy / n, dz / n
        cx, cy, cz = R_SiC * ux, R_SiC * uy, R_SiC * uz
        add("C", cx, cy, cz)
        ax, ay, az = -ux, -uy, -uz
        if abs(ax) < 0.9:
            px, py, pz = 1.0, 0.0, 0.0
        else:
            px, py, pz = 0.0, 1.0, 0.0
        vx, vy, vz = (
            px - (ax * px + ay * py + az * pz) * ax,
            py - (ax * px + ay * py + az * pz) * ay,
            pz - (ax * px + ay * py + az * pz) * az,
        )
        vn = (vx * vx + vy * vy + vz * vz) ** 0.5
        vx, vy, vz = vx / vn, vy / vn, vz / vn
        wx, wy, wz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
        for ang in (0.0, 2.0943951023931953, 4.1887902047863905):
            hx = cx + R_CH * (0.6 * ax + 0.8 * (math.cos(ang) * vx + math.sin(ang) * wx))
            hy = cy + R_CH * (0.6 * ay + 0.8 * (math.cos(ang) * vy + math.sin(ang) * wy))
            hz = cz + R_CH * (0.6 * az + 0.8 * (math.cos(ang) * vz + math.sin(ang) * wz))
            add("H", hx, hy, hz)

    return symbols, np.array(coords, dtype=float)


@lru_cache(maxsize=1)
def _tms_ref_sigma(xc: str) -> Dict[str, float]:
    symbols, coords_A = _tms_geometry()
    mol = _make_mol(symbols, coords_A, charge=0, spin=0)
    mf = _build_rks_or_uks(mol, xc=xc)
    _ = mf.kernel()
    tensors = _nmr_tensors_from_mf(mf)
    sigma = (np.trace(tensors, axis1=1, axis2=2) / 3.0).astype(float) if isinstance(tensors,
                                                                                    np.ndarray) and tensors.ndim == 3 else np.asarray(
        tensors, dtype=float)
    return {
        "H": float(np.mean(sigma[[i for i, s in enumerate(symbols) if s.upper() == "H"]])),
        "C": float(np.mean(sigma[[i for i, s in enumerate(symbols) if s.upper() == "C"]])),
    }


# ── Energetics for Boltzmann weights ─────────────────────────────────────────
def _single_point_energy_pcm(
        symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int, xc: str, solvent_eps: Optional[float]
) -> float:
    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)
    mf = _attach_pcm_and_build_scf(mol, xc=xc, solvent_eps=solvent_eps, use_pcm=True)
    e = float(mf.kernel())
    LOG.debug("PCM single-point energy (Ha): %.10f", e)
    return e


def _boltzmann_weights_from_energies(E_hartree: Sequence[float], temp_K: float) -> np.ndarray:
    kB_Ha_per_K = 3.166811563e-6
    E = np.array(E_hartree, dtype=float)
    Emin = float(np.min(E))
    beta = 1.0 / (kB_Ha_per_K * temp_K)
    x = -beta * (E - Emin)
    x -= np.max(x)
    w = np.exp(x)
    return (w / np.sum(w)).astype(float)


# ── I/O: clusters & representatives ──────────────────────────────────────────
@dataclass
class ClusterRow:
    cid: int
    fraction: float
    rep_pdb: Path


def _guess_rep_path(tag: str, cid: int) -> Path:
    p = CLUSTERS_DIR / f"{tag}_cluster_{cid}_rep.pdb"
    if p.exists():
        return p
    for name in (
            f"{tag}_cluster{cid}_rep.pdb",
            f"{tag}_c{cid}_rep.pdb",
            f"{tag}_cluster_{cid}_representative.pdb",
            f"{tag}_cluster_{cid}.pdb",
    ):
        q = CLUSTERS_DIR / name
        if q.exists():
            return q
    for pat in (f"{tag}_cluster_{cid}_*rep*.pdb", f"{tag}*cluster*{cid}*rep*.pdb"):
        for q in CLUSTERS_DIR.glob(pat):
            return q
    examples = sorted(CLUSTERS_DIR.glob(f"{tag}_cluster_*_rep.pdb"))[:10]
    hint = "\n".join(f"  - {x.name}" for x in examples) or "  (no matching files)"
    raise FileNotFoundError(
        f"No rep PDB found for tag={tag} cid={cid}\n"
        f"Looked for: {tag}_cluster_{cid}_rep.pdb\n"
        f"Nearby examples:\n{hint}"
    )


# ── TSV loader (pandas if available; else csv) ───────────────────────────────
def _load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    CID_COLS = ["cid", "cluster_id", "cluster"]
    FRAC_COLS = ["fraction", "weight", "pop", "pop_frac"]
    REP_COLS = ["rep", "pdb", "path", "rep_path", "representative", "medoid", "medoid_path"]

    def _norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    def _pick(series_names: List[str], candidates: List[str]) -> Optional[str]:
        sset = {_norm(x) for x in series_names}
        for c in candidates:
            if _norm(c) in sset:
                for x in series_names:
                    if _norm(x) == _norm(c):
                        return x
        return None

    rows: List[ClusterRow] = []

    if _HAS_PANDAS:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
        cols = list(df.columns)
        cid_name = _pick(cols, CID_COLS)
        frac_name = _pick(cols, FRAC_COLS)
        rep_name = _pick(cols, REP_COLS)

        if cid_name is None:
            # Treat as headerless: first two columns are cid, fraction
            df = pd.read_csv(path, sep="\t", comment="#", header=None, dtype=str, keep_default_na=False)
            df = df.dropna(how="all")
            if df.shape[1] < 2:
                raise ValueError(f"[{path.name}] Need at least two columns (cid, fraction).")
            df.columns = ["cid", "fraction"] + [f"extra_{i}" for i in range(df.shape[1] - 2)]
            cid_name, frac_name, rep_name = "cid", "fraction", None
        else:
            ren = {}
            ren[cid_name] = "cid"
            if frac_name:
                ren[frac_name] = "fraction"
            if rep_name:
                ren[rep_name] = "rep_path"
            if ren:
                df = df.rename(columns=ren)

        # Coerce types
        try:
            cid_vals = pd.to_numeric(df["cid"], errors="raise")
        except Exception as e:
            bad = df["cid"].iloc[0] if not df.empty else "<empty>"
            raise ValueError(f"[{path.name}] Non-numeric cid value like '{bad}'.") from e

        frac_vals = None
        if "fraction" in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frac_vals = pd.to_numeric(df["fraction"], errors="coerce")

        rep_vals = df["rep_path"] if "rep_path" in df.columns else None

        for i in range(len(df)):
            cid = int(cid_vals.iloc[i])
            frac = float(frac_vals.iloc[i]) if (frac_vals is not None and pd.notna(frac_vals.iloc[i])) else float("nan")
            if rep_vals is not None and isinstance(rep_vals.iloc[i], str) and rep_vals.iloc[i].strip():
                rp = rep_vals.iloc[i].strip()
                rep = Path(rp).resolve() if Path(rp).is_absolute() else (path.parent / rp).resolve()
            else:
                rep = _guess_rep_path(tag, cid)
            rows.append(ClusterRow(cid=cid, fraction=frac, rep_pdb=rep))

    else:
        with path.open("r", newline="") as fh:
            r = csv.reader(fh, delimiter="\t")
            for ln in r:
                if not ln or ln[0].lstrip().startswith("#"):
                    continue
                # assume headerless: cid, fraction[, rep]
                try:
                    cid = int(float(ln[0]))
                except Exception as e:
                    raise ValueError(f"[{path.name}] Expected numeric cid in first column, got {ln[0]!r}") from e
                frac = float(ln[1]) if len(ln) > 1 and ln[1] != "" else float("nan")
                rep = Path(ln[2]).resolve() if len(ln) > 2 and ln[2] != "" else _guess_rep_path(tag, cid)
                if not rep.is_absolute():
                    rep = (path.parent / rep).resolve()
                rows.append(ClusterRow(cid=cid, fraction=frac, rep_pdb=rep))

    # Equal-weight missing/NaN fractions
    fracs = np.array([row.fraction for row in rows], dtype=float)
    if not np.all(np.isfinite(fracs)):
        n = len(rows)
        for row in rows:
            row.fraction = 1.0 / n

    return rows


# ── Minimal PDB reader ───────────────────────────────────────────────────────
def _read_pdb_atoms(pdb_path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    atom_names: List[str] = []
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    with pdb_path.open("r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                el = line[76:78].strip() or name[0]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_names.append(name)
                symbols.append(el)
                coords.append((x, y, z))
    if not atom_names:
        raise RuntimeError(f"No atoms parsed from {pdb_path}")
    return atom_names, symbols, np.array(coords, dtype=float)


# ── Charge/spin per tag ──────────────────────────────────────────────────────
def _get_charge_spin_for_tag(tag: str) -> Tuple[int, int]:
    t = tag.lower()
    if "deprot" in t or "anion" in t:
        return (-1, 0)
    if "cation" in t or "prot" in t:
        return (+1, 0)
    return (0, 0)


# ── Writers ──────────────────────────────────────────────────────────────────
def _write_cluster_shifts(
        out_dir: Path, tag: str, cid: int, atom_names: Sequence[str], atom_symbols: Sequence[str],
        sigma_iso: np.ndarray, delta_ppm: np.ndarray
) -> Path:
    out_path = out_dir / f"{tag}_cluster_{cid}_shifts.tsv"
    with out_path.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tsigma_iso\tshift_ppm\n")
        for i, (nm, el, sig, dppm) in enumerate(zip(atom_names, atom_symbols, sigma_iso, delta_ppm)):
            fh.write(f"{i}\t{nm}\t{el}\t{_fmt(sig)}\t{_fmt(dppm)}\n")
    return out_path


def _write_fastavg_shifts(
        out_dir: Path, tag: str, atom_names: Sequence[str], atom_symbols: Sequence[str], delta_ppm: np.ndarray
) -> Path:
    out_path = out_dir / f"{tag}_fastavg_shifts.tsv"
    with out_path.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tshift_ppm\n")
        for i, (nm, el, dppm) in enumerate(zip(atom_names, atom_symbols, delta_ppm)):
            fh.write(f"{i}\t{nm}\t{el}\t{_fmt(dppm)}\n")
    return out_path


def _write_params(out_dir: Path, tag: str, params: Dict[str, str | float | int | bool]) -> Path:
    out_path = out_dir / f"{tag}_params.txt"
    with out_path.open("w") as fh:
        for k, v in params.items():
            fh.write(f"{k}: {v}\n")
    return out_path


# ── Core per-tag processing ──────────────────────────────────────────────────
def _process_tag(
        tag: str,
        *,
        xc: str,
        basis: str,
        solvent_eps: Optional[float],
        do_geom_opt: bool,
        use_boltz: bool,
        temp_K: float,
        opt_backend: str,
        quiet: bool,
) -> None:
    global BASIS_DEFAULT
    BASIS_DEFAULT = basis

    LOG.info("[tag] %s", tag)

    cluster_tsv = CLUSTERS_DIR / f"{tag}_clusters.tsv"
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Missing cluster table: {cluster_tsv}")

    rows = _load_clusters_table(cluster_tsv, tag=tag)
    (charge, spin) = _get_charge_spin_for_tag(tag)
    ref_sigma = _tms_ref_sigma(xc=xc)

    per_cluster_delta: List[np.ndarray] = []
    per_cluster_frac: List[float] = []
    per_cluster_energy: List[float] = []
    atom_names_ref: Optional[List[str]] = None
    atom_symbols_ref: Optional[List[str]] = None

    for row in rows:
        atom_names, symbols, coords_A = _read_pdb_atoms(row.rep_pdb)
        if do_geom_opt:
            coords_A = _optimize_geometry_pcm(
                symbols, coords_A, charge=charge, spin=spin, xc=xc, solvent_eps=solvent_eps,
                backend=opt_backend, quiet=quiet
            )

        sigma_iso = _compute_sigma_iso(symbols, coords_A, charge=charge, spin=spin, xc=xc)
        delta_ppm = _sigma_to_delta_ppm(symbols, sigma_iso, ref_sigma)

        _ = _write_cluster_shifts(OUT_DIR, tag, row.cid, atom_names, symbols, sigma_iso, delta_ppm)

        per_cluster_delta.append(delta_ppm)
        per_cluster_frac.append(float(row.fraction))

        if use_boltz:
            e_pcm = _single_point_energy_pcm(
                symbols, coords_A, charge=charge, spin=spin, xc=xc, solvent_eps=solvent_eps
            )
            per_cluster_energy.append(e_pcm)

        if atom_names_ref is None:
            atom_names_ref = list(atom_names)
            atom_symbols_ref = list(symbols)

    if use_boltz:
        weights = _boltzmann_weights_from_energies(per_cluster_energy, temp_K=temp_K)
    else:
        fracs = np.array(per_cluster_frac, dtype=float)
        denom = float(np.sum(fracs))
        weights = (fracs / denom) if denom > 1e-15 else (np.ones_like(fracs) / fracs.size)

    all_delta = np.stack(per_cluster_delta, axis=0)
    fastavg = np.sum(all_delta * weights[:, None], axis=0)

    _ = _write_fastavg_shifts(OUT_DIR, tag, atom_names_ref, atom_symbols_ref, fastavg)
    _ = _write_params(
        OUT_DIR,
        tag,
        {
            "xc": xc,
            "basis": basis,
            "charge": charge,
            "spin": spin,
            "pcm_eps": (solvent_eps if solvent_eps is not None else "None"),
            "geom_optimized": do_geom_opt,
            "opt_backend": opt_backend,
            "weights": ("Boltzmann" if use_boltz else "MD fractions"),
            "temperature_K": temp_K,
            "k_B[Hartree/K]": 3.166811563e-6,
            "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        },
    )
    LOG.info("[ok] %s: %d clusters → fastavg written.", tag, len(rows))


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute NMR shifts (fast-exchange avg) from clustered conformers.")
    p.add_argument("--temp", type=float, default=TEMPERATURE_K_DEFAULT, help="Temperature [K] for Boltzmann weights.")
    p.add_argument("--eps", type=float, default=PCM_EPS_DEFAULT, help="PCM dielectric (e.g., 46.7 for DMSO).")
    p.add_argument("--no-pcm", action="store_true", help="Disable PCM (for geom energy and single-point E).")
    p.add_argument("--no-opt", action="store_true", help="Skip geometry optimization of reps.")
    p.add_argument("--opt", choices=["auto", "geom", "berny", "none"], default="auto",
                   help="Geometry optimization backend.")
    p.add_argument("--no-boltz", action="store_true", help="Use MD fractions instead of Boltzmann weights.")
    p.add_argument("--xc", type=str, default=DFT_XC_DEFAULT, help="DFT functional (default b3lyp).")
    p.add_argument("--basis", type=str, default=BASIS_DEFAULT, help="Gaussian basis (default def2-tzvp).")
    p.add_argument("--tags", type=str, nargs="*", help="Explicit tags (default: all *_clusters.tsv).")
    p.add_argument("--log-level", type=str, default="INFO",
                   help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    p.add_argument("--quiet", action="store_true", help="Reduce informational messages.")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    _setup_logging(args["log_level"] if isinstance(args, dict) else args.log_level, args.quiet)

    _mkdir_p(OUT_DIR)

    solvent_eps = None if args.no_pcm else float(args.eps)
    do_geom_opt = (not args.no_opt)
    use_boltz = (not args.no_boltz)

    # Environment banner & GPU
    gpu = _detect_gpu()
    LOG.info("PySCF %s | DFT=%s | basis=%s | PCM=%s | opt=%s | temp=%.2f K",
             getattr(pyscf, "__version__", "unknown"),
             args.xc, args.basis,
             ("off" if solvent_eps is None else f"ddCOSMO eps={solvent_eps:.2f}"),
             ("none" if not do_geom_opt else args.opt),
             float(args.temp))
    if gpu["gpu_available"]:
        LOG.info("GPU detected: backends=%s | devices=%s", ",".join(gpu["backend"]), gpu["devices"])
    else:
        LOG.info("GPU not detected; running on CPU.")
    if gpu["env"]:
        LOG.debug("Env: %s", gpu["env"])

    if args.tags:
        tags = args.tags
    else:
        cluster_files = sorted(CLUSTERS_DIR.glob("*_clusters.tsv"))
        if not cluster_files:
            LOG.error("No *_clusters.tsv in %s/", CLUSTERS_DIR)
            return
        tags = [cf.stem.removesuffix("_clusters") for cf in cluster_files]

    for tag in tags:
        _process_tag(
            tag,
            xc=args.xc,
            basis=args.basis,
            solvent_eps=solvent_eps,
            do_geom_opt=do_geom_opt,
            use_boltz=use_boltz,
            temp_K=float(args.temp),
            opt_backend=args.opt,
            quiet=args.quiet,
        )
    LOG.info("[done]")


if __name__ == "__main__":
    main()
