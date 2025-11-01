#!/usr/bin/env python3
# f_predict_shifts_core.py
# Core utilities for NMR shift prediction
# Shared by: f_predict_shifts_compute.py and f_predict_shifts_average.py
# 2025-11-01
"""
Core utilities for predicting NMR observables from clustered conformers.

This module:
  • parses cluster tables / representative PDBs
  • builds and runs (U)KS DFT with PySCF (gpu4pyscf if available)
  • extracts isotropic nuclear shieldings σ_iso → chemical shifts δ (ppm)
  • extracts scalar spin–spin J couplings (Hz) for NMR-active nuclei
  • computes ddCOSMO single-point energies for Boltzmann weights
  • writes per-cluster .tsv outputs used by the pipeline
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyscf import gto, dft

# Silence PySCF "under testing" noise, keep real errors visible
warnings.filterwarnings("ignore", message=r"Module .* is under testing", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Module .* is not fully tested", category=UserWarning)

LOG = logging.getLogger("nmrshifts.core")

# Optional GPU backend
try:
    from gpu4pyscf import dft as g4dft  # type: ignore

    GPU4PYSCF_AVAILABLE = True
except Exception:
    GPU4PYSCF_AVAILABLE = False

# Optional Berny (geometry optimizer)
try:
    import berny  # type: ignore

    BERNY_AVAILABLE = True
except Exception:
    BERNY_AVAILABLE = False

# Paths / defaults
OUT_DIR = Path("f_predict_shifts")
CLUSTERS_DIR = Path("e_cluster")
DFT_XC_DEFAULT = "b3lyp"
BASIS_DEFAULT = "def2-tzvp"
SCF_MAXCYC = 200
SCF_CONV_TOL = 1e-9
GRAD_CONV_TOL = 3e-4

# Solvent map (eps ~298 K). Override with --eps
SOLVENT_EPS: Dict[str, float] = {
    "vacuum": 1.0,
    "water": 78.36, "h2o": 78.36,
    "dmso": 46.7,
    "meoh": 32.7, "methanol": 32.7,
    "mecn": 35.7, "acetonitrile": 35.7,
    "chcl3": 4.81, "cdcl3": 4.81, "chloroform": 4.81,
    "thf": 7.58,
    "toluene": 2.38,
    "acetone": 20.7,
}


@dataclass
class ClusterRow:
    cid: int
    fraction: float
    rep_pdb: Path


def setup_logging(level: str = "INFO", quiet: bool = False) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    if quiet:
        numeric = max(numeric, logging.WARNING)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pyscf").setLevel(logging.WARNING)


def detect_gpu() -> Dict[str, object]:
    info = {"gpu4pyscf": GPU4PYSCF_AVAILABLE, "cupy": False, "devices": []}
    try:
        import cupy as cp  # type: ignore
        n = int(cp.cuda.runtime.getDeviceCount())
        info["cupy"] = n > 0
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            info["devices"].append(str(name))
    except Exception:
        pass
    return info


def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fmt(x: float) -> str:
    return f"{x:.6f}" if np.isfinite(x) else "nan"


def get_charge_spin(tag: str) -> Tuple[int, int]:
    t = tag.lower()
    if "deprot" in t or "anion" in t:
        return (-1, 0)
    if "cation" in t or "prot" in t:
        return (+1, 0)
    return (0, 0)


def guess_rep_path(tag: str, cid: int) -> Path:
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
    # final fallback: <tag>_cluster_<cid>_rep.pdb even if it doesn't exist
    return p


def load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    """
    Expected columns (loose):
      - cluster id (cid or cluster or cluster_id or similar)
      - optional fraction / population / weight
      - optional rep pdb path

    We make best-effort guesses for column names.
    """
    df = pd.read_csv(path, sep=None, engine="python", comment="#").copy()

    cols = {c.lower(): c for c in df.columns}
    cid = None
    frac = None
    rep = None

    for cand in ("cid", "cluster", "cluster_id", "id"):
        if cand in cols:
            cid = cols[cand]
            break
    for cand in ("fraction", "pop", "population", "weight", "boltz", "md_fraction"):
        if cand in cols:
            frac = cols[cand]
            break
    for cand in ("rep", "rep_pdb", "pdb", "representative"):
        if cand in cols:
            rep = cols[cand]
            break

    if cid is None:
        raise ValueError(f"[{path}] no cluster id column found.")

    # normalize names
    if cid == "cid" and (frac is None and rep is None):
        # probably already [cid,fraction,rep_path]
        if df.shape[1] < 2:
            raise ValueError(f"[{path.name}] Need at least two columns (cid, fraction).")
        df.columns = ["cid", "fraction"] + [f"x{i}" for i in range(df.shape[1] - 2)]
        cid, frac, rep = "cid", "fraction", None
    else:
        ren = {cid: "cid"}
        if frac: ren[frac] = "fraction"
        if rep:  ren[rep] = "rep_path"
        df = df.rename(columns=ren)

    cidv = pd.to_numeric(df["cid"], errors="raise")
    frv = pd.to_numeric(df["fraction"], errors="coerce") if "fraction" in df.columns else None
    rpv = df["rep_path"] if "rep_path" in df.columns else None

    rows: List[ClusterRow] = []
    for i in range(len(df)):
        cid_i = int(cidv.iloc[i])
        if frv is not None and pd.notna(frv.iloc[i]):
            frac_i = float(frv.iloc[i])
        else:
            frac_i = float("nan")
        if rpv is not None and isinstance(rpv.iloc[i], str) and rpv.iloc[i].strip():
            rp = Path(rpv.iloc[i])
            rep_path = rp if rp.is_absolute() else (path.parent / rp)
        else:
            rep_path = guess_rep_path(tag, cid_i)
        rows.append(ClusterRow(cid=cid_i, fraction=frac_i, rep_pdb=rep_path.resolve()))

    fracs = np.array([r.fraction for r in rows], dtype=float)
    if not np.all(np.isfinite(fracs)):
        n = len(rows)
        for r in rows:
            r.fraction = 1.0 / n
    return rows


def read_pdb_atoms(pdb: Path) -> Tuple[List[str], List[str], np.ndarray]:
    names, syms, coords = [], [], []
    with pdb.open("r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                el = line[76:78].strip() or name[0]
                x = float(line[30:38]);
                y = float(line[38:46]);
                z = float(line[46:54])
                names.append(name);
                syms.append(el);
                coords.append((x, y, z))
    return names, syms, np.array(coords, dtype=float)


def make_mol(symbols: Sequence[str],
             coords_A: np.ndarray,
             charge: int,
             spin: int,
             basis: str) -> gto.Mole:
    mol = gto.M(
        atom=[(s, tuple(xyz)) for (s, xyz) in zip(symbols, coords_A)],
        charge=charge,
        spin=spin,
        basis=basis,
        unit="Angstrom",
        verbose=0,
    )
    return mol


def build_scf(mol: gto.Mole, xc: str, *, use_gpu: bool):
    if use_gpu and GPU4PYSCF_AVAILABLE:
        mf = g4dft.RKS(mol) if mol.spin == 0 else g4dft.UKS(mol)
        try:
            mf = mf.to_gpu()
            LOG.debug("SCF backend: %s (GPU)", mf.__class__)
        except Exception as e:
            LOG.warning("gpu4pyscf to_gpu failed: %s; CPU fallback.", e)
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            LOG.debug("SCF backend: %s (CPU fallback)", mf.__class__)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        LOG.debug("SCF backend: %s (CPU)", mf.__class__)
    mf.xc = xc;
    mf.max_cycle = SCF_MAXCYC;
    mf.conv_tol = SCF_CONV_TOL
    return mf


def attach_pcm(mf, eps: Optional[float]):
    if eps is None:
        return mf
    try:
        from pyscf import solvent
        mf_pcm = solvent.ddCOSMO(mf)
        mf_pcm.with_solvent.eps = float(eps)
        LOG.debug("ddCOSMO attached (eps=%.3f)", eps)
        return mf_pcm
    except Exception as e:
        LOG.warning("PCM attach failed: %s; gas-phase.", e)
        return mf


def nmr_tensors_from_mf(mf):
    try:  # unified class
        from pyscf.prop.nmr import NMR  # type: ignore
        return NMR(mf).kernel()
    except Exception:
        pass
    if hasattr(mf, "NMR"):
        try:
            return mf.NMR().kernel()
        except Exception:
            pass
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
    raise RuntimeError("PySCF NMR interface not found (prop.nmr.NMR / mf.NMR / rks/uks/rhf/uhf).")


def _property_on_cpu(mol, gpu_mf, xc: str):
    """Build a CPU MF using the converged density from a GPU MF if available."""
    from pyscf import dft as _dft
    mf_cpu = _dft.RKS(mol) if mol.spin == 0 else _dft.UKS(mol)
    mf_cpu.xc = xc
    mf_cpu.max_cycle = SCF_MAXCYC
    mf_cpu.conv_tol = SCF_CONV_TOL
    # Try to reuse density from GPU object
    dm0 = None
    try:
        dm0 = gpu_mf.make_rdm1()
        # CuPy array -> NumPy
        if hasattr(dm0, "get"):
            dm0 = dm0.get()
    except Exception:
        dm0 = None
    try:
        if dm0 is not None:
            mf_cpu.kernel(dm0=dm0)
        else:
            mf_cpu.kernel()
    except Exception:
        # last resort
        mf_cpu.kernel()
    return mf_cpu


def compute_sigma_iso(symbols, coords_A, charge, spin, xc, basis, use_gpu: bool):
    mol = make_mol(symbols, coords_A, charge, spin, basis)
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()

    # Try NMR on the current MF (works for CPU MF; usually fails for gpu4pyscf)
    try:
        arr = np.asarray(nmr_tensors_from_mf(mf))
    except Exception:
        LOG.debug("NMR not available on %s; falling back to CPU MF for property.", mf.__class__)
        mf_cpu = _property_on_cpu(mol, mf, xc)
        arr = np.asarray(nmr_tensors_from_mf(mf_cpu))

    if arr.ndim == 3 and arr.shape[-1] == 3:
        return (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.mean(axis=1).astype(float)
    return np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)


def compute_spinspin_JHz(symbols,
                         coords_A,
                         charge,
                         spin,
                         xc,
                         basis,
                         use_gpu: bool,
                         isotopes_keep: Sequence[str] = ("1H", "13C")) -> Tuple[np.ndarray, List[str]]:
    """Return (J_sel_Hz, kept_labels).

    Steps:
      1. SCF (same DFT settings as shifts)
      2. PySCF spin–spin coupling (prop.ssc.SSC) → full J matrix in Hz
      3. Keep only NMR-active isotopes listed in `isotopes_keep`
         (default keeps 1H and 13C; add "15N", "19F", ... if needed)

    kept_labels[i] is like "H1", "H2", ... matching rows/cols of J_sel_Hz.
    """
    mol = make_mol(symbols, coords_A, charge, spin, basis)
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()

    # Try SSC on current MF (works for CPU MF; gpu4pyscf may fail)
    try:
        from pyscf.prop import ssc as _ssc  # type: ignore
        J_full = np.asarray(_ssc.SSC(mf).kernel(), dtype=float)
    except Exception:
        LOG.debug("SSC not available on %s; falling back to CPU MF for property.", mf.__class__)
        mf_cpu = _property_on_cpu(mol, mf, xc)
        from pyscf.prop import ssc as _ssc  # type: ignore
        J_full = np.asarray(_ssc.SSC(mf_cpu).kernel(), dtype=float)

    # Decide which nuclei to keep
    iso_map = {
        "H": "1H",
        "C": "13C",
        "N": "15N",
        "F": "19F",
        "P": "31P",
    }
    keep_idx: List[int] = []
    keep_labels: List[str] = []
    for i, el in enumerate(symbols):
        iso_guess = iso_map.get(el.capitalize())
        if iso_guess in isotopes_keep:
            keep_idx.append(i)
            keep_labels.append(f"{el}{i + 1}")

    if not keep_idx:
        return np.zeros((0, 0), dtype=float), []

    J_sel = J_full[np.ix_(keep_idx, keep_idx)].astype(float)
    return J_sel, keep_labels


def sigma_to_delta(symbols: Sequence[str], sigma_iso: np.ndarray, ref_sigma: Dict[str, float]) -> np.ndarray:
    out = np.full_like(sigma_iso, np.nan, dtype=float)
    for i, el in enumerate(symbols):
        elU = el.upper()
        if elU in ("H", "C"):
            out[i] = ref_sigma[elU] - sigma_iso[i]
    return out


def tms_geometry() -> Tuple[List[str], np.ndarray]:
    syms: List[str] = [];
    coords: List[Tuple[float, float, float]] = []

    def add(a, x, y, z):
        syms.append(a);
        coords.append((x, y, z))

    add("Si", 0.0, 0.0, 0.0)
    R_SiC, R_CH = 1.86, 1.09
    dirs = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
    for dx, dy, dz in dirs:
        n = (dx * dx + dy * dy + dz * dz) ** 0.5;
        ux, uy, uz = dx / n, dy / n, dz / n
        cx, cy, cz = R_SiC * ux, R_SiC * uy, R_SiC * uz;
        add("C", cx, cy, cz)
        ax, ay, az = -ux, -uy, -uz
        if abs(ax) < 0.9:
            px, py, pz = 1.0, 0.0, 0.0
        else:
            px, py, pz = 0.0, 1.0, 0.0
        vx, vy, vz = px - (ax * px + ay * py + az * pz) * ax, py - (ax * px + ay * py + az * pz) * ay, pz - (
                ax * px + ay * py + az * pz) * az
        vn = (vx * vx + vy * vy + vz * vz) ** 0.5;
        vx, vy, vz = vx / vn, vy / vn, vz / vn
        wx, wy, wz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
        for ang in (0.0, 2.0943951023931953, 4.1887902047863905):
            hx = cx + R_CH * (0.6 * ax + 0.8 * (math.cos(ang) * vx + math.sin(ang) * wx))
            hy = cy + R_CH * (0.6 * ay + 0.8 * (math.cos(ang) * vy + math.sin(ang) * wy))
            hz = cz + R_CH * (0.6 * az + 0.8 * (math.cos(ang) * vz + math.sin(ang) * wz))
            add("H", hx, hy, hz)
    return syms, np.array(coords, float)


@lru_cache(maxsize=1)
def tms_ref_sigma(xc: str, basis: str) -> Dict[str, float]:
    """
    Compute reference σ_iso for TMS once and cache it.
    We'll average over all equivalent H and C nuclei.
    """
    syms, coords_A = tms_geometry()
    mol = make_mol(syms, coords_A, charge=0, spin=0, basis=basis)
    mf = build_scf(mol, xc, use_gpu=False)
    _ = mf.kernel()
    arr = np.asarray(nmr_tensors_from_mf(mf))
    if arr.ndim == 3:
        sigma = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        sigma = arr.astype(float)
    else:
        sigma = np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)
    H = float(np.mean(sigma[[i for i, s in enumerate(syms) if s.upper() == "H"]]))
    C = float(np.mean(sigma[[i for i, s in enumerate(syms) if s.upper() == "C"]]))
    return {"H": H, "C": C}


def sp_energy_pcm(symbols,
                  coords_A,
                  charge,
                  spin,
                  xc,
                  basis,
                  eps: Optional[float],
                  use_gpu: bool) -> float:
    """
    Single-point energy in Hartree, possibly ddCOSMO-solvated.
    We only use this for Boltzmann weights (relative energies).
    """
    mol = make_mol(symbols, coords_A, charge, spin, basis)
    mf0 = build_scf(mol, xc, use_gpu=use_gpu)
    mf_pcm = attach_pcm(mf0, eps=eps)
    e_tot = mf_pcm.kernel()
    return float(e_tot)


def boltzmann_weights(E_hartree: Sequence[float], T_K: float) -> np.ndarray:
    """
    Return normalized Boltzmann weights at temperature T_K
    from a list of conformer energies in Hartree.
    """
    kB_kcal_per_K = 0.0019872041  # Boltzmann in kcal/(mol*K)
    hartree_to_kcalmol = 627.509474
    kB_Ha_per_K = kB_kcal_per_K / hartree_to_kcalmol

    E = np.array(E_hartree, dtype=float)
    Emin = float(np.min(E))
    beta = 1.0 / (kB_Ha_per_K * T_K)
    x = -beta * (E - Emin)
    x -= np.max(x)
    w = np.exp(x)
    return (w / np.sum(w)).astype(float)


def average_J_matrices(J_mats: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Return weighted average of J-coupling matrices.

    All J_mats[k] must have the same (M,M) shape corresponding to the
    same nucleus ordering / labels.  weights[k] can be any non-negative
    numbers (Boltzmann, MD fractions, ...).  We normalize them here.
    """
    if not J_mats:
        return np.zeros((0, 0), dtype=float)

    w = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(w)):
        raise ValueError("non-finite weights in average_J_matrices")

    w = w / np.sum(w)
    acc = np.zeros_like(J_mats[0], dtype=float)
    for wi, Ji in zip(w, J_mats):
        acc += wi * Ji
    return acc


def write_cluster_shifts(out_dir: Path, tag: str, cid: int, atom_names, atom_symbols, sigma_iso, delta_ppm) -> Path:
    out = out_dir / tag / f"cluster_{cid}_shifts.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tsigma_iso\tshift_ppm\n")
        for i, (nm, el, sig, dppm) in enumerate(zip(atom_names, atom_symbols, sigma_iso, delta_ppm)):
            fh.write(f"{i}\t{nm}\t{el}\t{fmt(sig)}\t{fmt(dppm)}\n")
    return out


def write_j_couplings(out_dir: Path,
                      tag: str,
                      cid: int,
                      labels: Sequence[str],
                      J_Hz: np.ndarray) -> Path:
    """Write per-cluster scalar couplings (Hz) as a tidy upper triangle.

    Columns:
        i,j,label_i,label_j,J_Hz
    Only i<j rows are written.
    """
    out = out_dir / tag / f"cluster_{cid}_j_couplings.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as fh:
        fh.write("# i\tj\tlabel_i\tlabel_j\tJ_Hz\n")
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                fh.write(
                    f"{i}\t{j}\t{labels[i]}\t{labels[j]}\t{fmt(float(J_Hz[i, j]))}\n"
                )
    return out


def write_params(out_dir: Path, tag: str, name: str, params: Dict[str, object]) -> Path:
    out = out_dir / tag / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for k, v in params.items():
            fh.write(f"{k}: {v}\n")
    return out
