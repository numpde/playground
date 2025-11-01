#!/usr/bin/env python3
# Core utilities for NMR shift prediction
# Shared by: f_predict_shifts_compute.py and f_predict_shifts_average.py
# 2025-11-01

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

# Optional optimizers
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
    for pat in (f"{tag}_cluster_{cid}_*rep*.pdb", f"{tag}*cluster*{cid}*rep*.pdb"):
        got = list(CLUSTERS_DIR.glob(pat))
        if got:
            return got[0]
    raise FileNotFoundError(f"No rep PDB found for tag={tag} cid={cid}")


def load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    CID = ["cid", "cluster_id", "cluster"]
    FRAC = ["fraction", "weight", "pop", "pop_frac"]
    REP = ["rep", "pdb", "path", "rep_path", "representative", "medoid", "medoid_path"]
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    cols = list(df.columns)

    def pick(cands: List[str]) -> Optional[str]:
        canon = {c.lower().replace(" ", "_") for c in cols}
        for k in cands:
            if k.lower().replace(" ", "_") in canon:
                for c in cols:
                    if c.lower().replace(" ", "_") == k.lower().replace(" ", "_"):
                        return c
        return None

    cid = pick(CID);
    frac = pick(FRAC);
    rep = pick(REP)
    if cid is None:
        df = pd.read_csv(path, sep="\t", comment="#", header=None, dtype=str, keep_default_na=False)
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
    if not names:
        raise RuntimeError(f"No atoms in {pdb}")
    return names, syms, np.array(coords, float)


def make_mol(symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int, basis: str) -> gto.Mole:
    mol = gto.Mole()
    mol.build(atom=[(s, tuple(r)) for s, r in zip(symbols, coords_A)],
              unit="Angstrom", basis=basis, charge=charge, spin=spin, verbose=0)
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
        syms.append(a); coords.append((x, y, z))

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


def sp_energy_pcm(symbols, coords_A, charge, spin, xc, basis, eps) -> float:
    mol = make_mol(symbols, coords_A, charge, spin, basis)
    mf = build_scf(mol, xc, use_gpu=False)  # keep PCM path consistent/CPU
    mf = attach_pcm(mf, eps)
    e = float(mf.kernel())
    LOG.debug("PCM single-point energy (Ha): %.10f | MF=%s", e, mf.__class__)
    return e


def boltzmann_weights(E_hartree: Sequence[float], T_K: float) -> np.ndarray:
    kB_Ha_per_K = 3.166811563e-6
    E = np.array(E_hartree, dtype=float)
    Emin = float(np.min(E))
    beta = 1.0 / (kB_Ha_per_K * T_K)
    x = -beta * (E - Emin)
    x -= np.max(x)
    w = np.exp(x)
    return (w / np.sum(w)).astype(float)


def write_cluster_shifts(out_dir: Path, tag: str, cid: int, atom_names, atom_symbols, sigma_iso, delta_ppm) -> Path:
    out = out_dir / tag / f"cluster_{cid}_shifts.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tsigma_iso\tshift_ppm\n")
        for i, (nm, el, sig, dppm) in enumerate(zip(atom_names, atom_symbols, sigma_iso, delta_ppm)):
            fh.write(f"{i}\t{nm}\t{el}\t{fmt(sig)}\t{fmt(dppm)}\n")
    return out


def write_params(out_dir: Path, tag: str, name: str, params: Dict[str, object]) -> Path:
    out = out_dir / tag / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for k, v in params.items():
            fh.write(f"{k}: {v}\n")
    return out
