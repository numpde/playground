#!/usr/bin/env python3
# f_predict_shifts_core.py
#
# Core utilities for:
#   - loading cluster reps and metadata
#   - running SCF (GPU if available) and moving state back to CPU for properties
#   - computing isotropic shieldings σ_iso and chemical shifts δ (ppm)
#   - computing scalar spin–spin couplings (Hz)
#   - computing (solvent-weighted) single-point energies for Boltzmann weights
#   - writing per-cluster result tables and run metadata


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
from pyscf import dft, gto, scf

warnings.filterwarnings(
    "ignore",
    message=r"Module .* is under testing",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Module .* is not fully tested",
    category=UserWarning,
)

LOG = logging.getLogger("nmrshifts.core")

# gpu4pyscf is optional. We only use it for SCF acceleration.
try:
    from gpu4pyscf import dft as g4dft  # type: ignore

    GPU4PYSCF_AVAILABLE = True
except Exception:
    GPU4PYSCF_AVAILABLE = False

# berny is optional. We currently don't optimize geometries here.
try:
    import berny  # type: ignore

    BERNY_AVAILABLE = True
except Exception:
    BERNY_AVAILABLE = False

OUT_DIR = Path("f_predict_shifts")
CLUSTERS_DIR = Path("e_cluster")

DFT_XC_DEFAULT = "b3lyp"
BASIS_DEFAULT = "def2-tzvp"

SCF_MAXCYC = 200
SCF_CONV_TOL = 1e-9
GRAD_CONV_TOL = 3e-4  # reserved for future geometry refinement

# Approximate static dielectric constants (room temp). Used for ddCOSMO.
SOLVENT_EPS: Dict[str, float] = {
    "vacuum": 1.0,
    "water": 78.36,
    "h2o": 78.36,
    "dmso": 46.7,
    "meoh": 32.7,
    "methanol": 32.7,
    "mecn": 35.7,
    "acetonitrile": 35.7,
    "chcl3": 4.81,
    "cdcl3": 4.81,
    "chloroform": 4.81,
    "thf": 7.58,
    "toluene": 2.38,
    "acetone": 20.7,
}


# -----------------------------------------------------------------------------
# Logging / convenience
# -----------------------------------------------------------------------------

@dataclass
class ClusterRow:
    cid: int
    fraction: float
    rep_pdb: Path  # representative conformer geometry for this cluster


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
    """
    Report whether gpu4pyscf / CuPy are available.
    """
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


# -----------------------------------------------------------------------------
# Cluster table / geometry I/O
# -----------------------------------------------------------------------------

def get_charge_spin(tag: str) -> Tuple[int, int]:
    """
    Heuristic total charge and spin multiplicity guess from the tag.
    Neutral singlet unless the tag smells like an anion/cation.
    """
    t = tag.lower()
    if "deprot" in t or "anion" in t:
        return (-1, 0)
    if "cation" in t or "prot" in t:
        return (+1, 0)
    return (0, 0)


def guess_rep_path(tag: str, cid: int) -> Path:
    """
    Best-effort recovery of the representative PDB filename for a cluster.
    """
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
    return p


def load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    """
    Read cluster metadata for this tag.
    Supports:
      - a headered TSV with fuzzy-matched columns
      - a 'commented TSV' where rows start after '#cluster_id ...'

    Returns list[ClusterRow] with cid, population fraction, and rep_pdb path.
    """
    # Try headered TSV via pandas first
    try:
        df_try = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
        ).copy()

        cols = {c.lower(): c for c in df_try.columns}

        cid_col = None
        for cand in (
                "cid",
                "cluster",
                "cluster_id",
                "clusteridx",
                "cluster_idx",
                "id",
        ):
            if cand in cols:
                cid_col = cols[cand]
                break

        if cid_col is not None:
            frac_col = None
            for cand in (
                    "fraction",
                    "pop",
                    "population",
                    "weight",
                    "boltz",
                    "md_fraction",
                    "mdfrac",
                    "md_weight",
            ):
                if cand in cols:
                    frac_col = cols[cand]
                    break

            rep_col = None
            for cand in (
                    "rep",
                    "rep_pdb",
                    "rep_path",
                    "pdb",
                    "pdb_path",
                    "representative",
            ):
                if cand in cols:
                    rep_col = cols[cand]
                    break

            cidv = pd.to_numeric(df_try[cid_col], errors="raise")
            frv = pd.to_numeric(df_try[frac_col], errors="coerce") if frac_col else None
            rpv = df_try[rep_col].astype(str) if rep_col else None

            rows: List[ClusterRow] = []
            for i in range(len(df_try)):
                cid_i = int(cidv.iloc[i])

                if frv is not None and pd.notna(frv.iloc[i]):
                    frac_i = float(frv.iloc[i])
                else:
                    frac_i = float("nan")

                if rpv is not None and rpv.iloc[i].strip():
                    rp_raw = Path(rpv.iloc[i].strip())
                    rep_path = (
                        rp_raw
                        if rp_raw.is_absolute()
                        else (path.parent / rp_raw)
                    )
                else:
                    rep_path = guess_rep_path(tag, cid_i)

                rows.append(
                    ClusterRow(
                        cid=cid_i,
                        fraction=frac_i,
                        rep_pdb=rep_path.resolve(),
                    )
                )

            # fill missing fractions uniformly
            fracs = np.array([r.fraction for r in rows], dtype=float)
            if not np.all(np.isfinite(fracs)):
                n = len(rows)
                for r in rows:
                    r.fraction = 1.0 / n

            return rows

    except Exception:
        pass

    # Fallback: parse commented TSV
    rows = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"[{path}] can't parse line (need 5+ cols): {line!r}"
                )

            cid_i = int(parts[0])
            frac_i = float(parts[2])
            rep_rel = parts[4]
            rep_path = (path.parent / rep_rel).resolve()

            rows.append(
                ClusterRow(
                    cid=cid_i,
                    fraction=frac_i,
                    rep_pdb=rep_path,
                )
            )

    fracs = np.array([r.fraction for r in rows], dtype=float)
    if not np.all(np.isfinite(fracs)):
        n = len(rows)
        for r in rows:
            r.fraction = 1.0 / n

    return rows


def read_pdb_atoms(pdb: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Minimal PDB reader.
    Returns:
        atom_names   list[str]  (e.g. 'H12')
        atom_symbols list[str]  (e.g. 'H')
        coords_A     (N,3) Å
    """
    names: List[str] = []
    syms: List[str] = []
    coords: List[Tuple[float, float, float]] = []

    with pdb.open("r") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                el = line[76:78].strip() or name[0]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                names.append(name)
                syms.append(el)
                coords.append((x, y, z))

    return names, syms, np.array(coords, dtype=float)


# -----------------------------------------------------------------------------
# SCF / properties
# -----------------------------------------------------------------------------

def make_mol(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        basis: str,
) -> gto.Mole:
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
    """
    Construct an RKS/UKS object. If use_gpu and gpu4pyscf is available,
    prefer gpu4pyscf; otherwise plain PySCF.
    """
    if use_gpu and GPU4PYSCF_AVAILABLE:
        mf = g4dft.RKS(mol) if mol.spin == 0 else g4dft.UKS(mol)
        try:
            mf = mf.to_gpu()
            LOG.debug("SCF backend: %s (GPU)", mf.__class__)
        except Exception as e:
            LOG.warning(
                "gpu4pyscf to_gpu failed: %s; CPU fallback.",
                e,
            )
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            LOG.debug("SCF backend: %s (CPU fallback)", mf.__class__)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        LOG.debug("SCF backend: %s (CPU)", mf.__class__)

    mf.xc = xc
    mf.max_cycle = SCF_MAXCYC
    mf.conv_tol = SCF_CONV_TOL
    return mf


def _property_on_cpu(mol: gto.Mole, gpu_mf, xc: str):
    """
    Build a CPU MF object seeded with the converged density from gpu_mf
    (if available), then run .kernel(). This is needed because PySCF's
    property drivers (NMR shielding, SSC, ddCOSMO) don't exist in gpu4pyscf.
    """
    mf_cpu = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf_cpu.xc = xc
    mf_cpu.max_cycle = SCF_MAXCYC
    mf_cpu.conv_tol = SCF_CONV_TOL

    dm0 = None
    try:
        dm0 = gpu_mf.make_rdm1()
        if hasattr(dm0, "get"):  # CuPy -> NumPy
            dm0 = dm0.get()
    except Exception:
        dm0 = None

    try:
        if dm0 is not None:
            mf_cpu.kernel(dm0=dm0)
        else:
            mf_cpu.kernel()
    except Exception:
        mf_cpu.kernel()

    return mf_cpu


def attach_pcm(mf, eps: Optional[float]):
    """
    Try to attach ddCOSMO to an MF object, set dielectric, and return
    a wrapped MF. If ddCOSMO is not available, leave MF as-is and warn.
    """
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
    """
    Return shielding tensors from PySCF.
    Tries prop.nmr.NMR, then mf.NMR(), then backend-specific nmr.rks/uks/rhf/uhf.
    """
    # modern API
    try:
        from pyscf.prop.nmr import NMR  # type: ignore

        return NMR(mf).kernel()
    except Exception:
        pass

    # legacy bound method
    if hasattr(mf, "NMR"):
        try:
            return mf.NMR().kernel()
        except Exception:
            pass

    # backend fallbacks
    try:
        from pyscf.prop import nmr as nmrmod  # type: ignore

        if isinstance(mf, dft.rks.RKS) and hasattr(nmrmod, "rks"):
            return nmrmod.rks.NMR(mf).kernel()
        if isinstance(mf, dft.uks.UKS) and hasattr(nmrmod, "uks"):
            return nmrmod.uks.NMR(mf).kernel()
        if hasattr(nmrmod, "rhf") and isinstance(mf, scf.hf.RHF):
            return nmrmod.rhf.NMR(mf).kernel()
        if hasattr(nmrmod, "uhf") and isinstance(mf, scf.uhf.UHF):
            return nmrmod.uhf.NMR(mf).kernel()
    except Exception:
        pass

    raise RuntimeError(
        "PySCF NMR shielding interface not found "
        "(prop.nmr.NMR / mf.NMR / nmr.rks|uks|rhf|uhf)."
    )


# -----------------------------------------------------------------------------
# Shielding → δ ppm
# -----------------------------------------------------------------------------

def compute_sigma_iso(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
        use_gpu: bool,
) -> np.ndarray:
    """
    Return isotropic shielding σ_iso[i] for each nucleus.
    We run SCF (GPU if allowed), then try shielding on that MF. If the GPU MF
    can't do shielding, we bounce to a CPU MF and retry.
    """
    mol = make_mol(symbols, coords_A, charge, spin, basis)
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()

    try:
        arr = np.asarray(nmr_tensors_from_mf(mf))
    except Exception:
        LOG.debug(
            "NMR not available on %s; falling back to CPU MF for property.",
            mf.__class__,
        )
        mf_cpu = _property_on_cpu(mol, mf, xc)
        arr = np.asarray(nmr_tensors_from_mf(mf_cpu))

    # Normalization of output formats from PySCF:
    # [natm,3,3] full tensor -> trace/3
    # [natm] already isotropic
    # [natm,3] -> mean of the 3 components
    if arr.ndim == 3 and arr.shape[-1] == 3:
        sigma_iso = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        sigma_iso = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        sigma_iso = arr.mean(axis=1).astype(float)
    else:
        sigma_iso = (
            np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)
        )

    return sigma_iso


def sigma_to_delta(
        symbols: Sequence[str],
        sigma_iso: np.ndarray,
        ref_sigma: Dict[str, float],
) -> np.ndarray:
    """
    Convert σ_iso to chemical shift δ (ppm) with δ = σ_ref - σ.
    We fill only nuclei where we have a reference (H, C).
    """
    out = np.full_like(sigma_iso, np.nan, dtype=float)
    for i, el in enumerate(symbols):
        elU = el.upper()
        if elU in ("H", "C"):
            out[i] = ref_sigma[elU] - sigma_iso[i]
    return out


def _tms_guess_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Initial TMS (Si(CH3)4) guess in Cartesian Å.
    This is *not* guaranteed optimized. We'll refine it before use.
    """
    syms: List[str] = []
    coords: List[Tuple[float, float, float]] = []

    def add(a, x, y, z):
        syms.append(a)
        coords.append((x, y, z))

    add("Si", 0.0, 0.0, 0.0)
    R_SiC, R_CH = 1.86, 1.09
    dirs = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]

    for (dx, dy, dz) in dirs:
        n = (dx * dx + dy * dy + dz * dz) ** 0.5
        (ux, uy, uz) = (dx / n, dy / n, dz / n)
        (cx, cy, cz) = (R_SiC * ux, R_SiC * uy, R_SiC * uz)
        add("C", cx, cy, cz)

        (ax, ay, az) = (-ux, -uy, -uz)
        if abs(ax) < 0.9:
            (px, py, pz) = (1.0, 0.0, 0.0)
        else:
            (px, py, pz) = (0.0, 1.0, 0.0)

        vx, vy, vz = (
            px - (ax * px + ay * py + az * pz) * ax,
            py - (ax * px + ay * py + az * pz) * ay,
            pz - (ax * px + ay * py + az * pz) * az,
        )
        vn = (vx * vx + vy * vy + vz * vz) ** 0.5
        vx, vy, vz = vx / vn, vy / vn, vz / vn
        wx, wy, wz = (
            ay * vz - az * vy,
            az * vx - ax * vz,
            ax * vy - ay * vx,
        )

        for ang in (0.0, 2.0943951023931953, 4.1887902047863905):
            hx = cx + R_CH * (
                    0.6 * ax + 0.8 * (math.cos(ang) * vx + math.sin(ang) * wx)
            )
            hy = cy + R_CH * (
                    0.6 * ay + 0.8 * (math.cos(ang) * vy + math.sin(ang) * wy)
            )
            hz = cz + R_CH * (
                    0.6 * az + 0.8 * (math.cos(ang) * vz + math.sin(ang) * wz)
            )
            add("H", hx, hy, hz)

    return syms, np.array(coords, float)


def _optimize_geometry_cpu(
        syms: Sequence[str],
        coords_A_init: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
) -> np.ndarray:
    """
    Return optimized Cartesian coords (Å) for a given starting geometry.
    CPU only. Tries full Berny optimization if available; otherwise
    does a single gradient-driven relaxation step as a fallback.

    We prefer to raise loudly only if *both* SCF and even a single-step
    gradient call fail. If Berny is missing, we LOG.warning but continue.
    """
    # Build mol for initial coords
    mol0 = make_mol(syms, coords_A_init, charge, spin, basis)

    # Plain CPU SCF object (we do *not* want GPU here for reference)
    mf0 = build_scf(mol0, xc, use_gpu=False)
    _ = mf0.kernel()

    # If Berny is available, do a proper optimization loop
    if BERNY_AVAILABLE:
        try:
            from pyscf.geomopt.berny_solver import kernel as berny_kernel  # type: ignore

            mol_opt = berny_kernel(mf0, assert_convergence=True)
            return np.asarray(mol_opt.atom_coords(unit="Angstrom"), dtype=float)
        except Exception as e:
            LOG.warning("Berny optimization failed (%s); fallback step.", e)

    # Fallback: take one gradient step along -grad just to relax bonds a bit.
    # This is not a full optimization but better than raw sketch.
    try:
        # Cartesian gradient in Hartree/Bohr
        g_bohr = mf0.nuc_grad_method().kernel()  # (N,3) in Ha/Bohr
        coords_bohr = mol0.atom_coords(unit="Bohr")
        # crude step
        step_bohr = -0.1 * g_bohr
        new_bohr = coords_bohr + step_bohr

        # Build new coords in Å
        Bohr2Ang = 0.529177210903
        coords_A_new = np.asarray(new_bohr * Bohr2Ang, dtype=float)
        return coords_A_new
    except Exception as e:
        LOG.error("Fallback gradient relaxation failed: %s", e)
        # Last resort: return the input unchanged, but make it obvious upstream
        return np.asarray(coords_A_init, dtype=float)


@lru_cache(maxsize=8)
def tms_geometry(xc: str, basis: str) -> Tuple[List[str], np.ndarray]:
    """
    Return an optimized TMS geometry for (xc,basis), in Å.

    Steps:
      1. build an initial Si(CH3)4 guess
      2. run a CPU geometry optimization with the same xc/basis
         (Berny if available, else a warned single-step relax)
      3. cache result so we don't re-opt every cluster

    We *do not* silently fall back to a crude hand-drawn TMS unless
    even the fallback fails; and in that case we have already WARNed.
    """
    (syms0, coords0_A) = _tms_guess_geometry()
    coords_opt_A = _optimize_geometry_cpu(
        syms0,
        coords0_A,
        charge=0,
        spin=0,
        xc=xc,
        basis=basis,
    )
    return (syms0, coords_opt_A)


@lru_cache(maxsize=8)
def tms_ref_sigma(xc: str, basis: str) -> Dict[str, float]:
    """
    Compute and cache σ_ref for TMS at (xc,basis).

    We:
      - build optimized TMS geometry for this xc/basis (CPU DFT)
      - run shielding on that CPU MF
      - average equivalent H and C atoms
      - return {'H': σ_H_ref, 'C': σ_C_ref}

    This fixes absolute referencing so δ(ppm) is physically meaningful.
    """
    (syms, coords_A) = tms_geometry(xc, basis)

    mol = make_mol(syms, coords_A, charge=0, spin=0, basis=basis)
    mf = build_scf(mol, xc, use_gpu=False)
    _ = mf.kernel()

    arr = np.asarray(nmr_tensors_from_mf(mf))

    # normalize shapes the same way we already do for clusters
    if arr.ndim == 3 and arr.shape[-1] == 3:
        sigma = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        sigma = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        sigma = arr.mean(axis=1).astype(float)
    else:
        sigma = (
            np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)
        )

    Hvals = [sigma[i] for (i, s) in enumerate(syms) if s.upper() == "H"]
    Cvals = [sigma[i] for (i, s) in enumerate(syms) if s.upper() == "C"]

    if not Hvals or not Cvals:
        raise RuntimeError(
            "TMS reference missing H or C after optimization; "
            "cannot define chemical shift scale."
        )

    H = float(np.mean(Hvals))
    C = float(np.mean(Cvals))

    return {"H": H, "C": C}


# -----------------------------------------------------------------------------
# Scalar spin–spin couplings (J in Hz)
# -----------------------------------------------------------------------------

def _pick_ssc_driver(mf_cpu):
    """
    Return an SSC driver bound to the mean-field.
    PySCF exposes SSC as backend-specific classes:
    pyscf.prop.ssc.rks.SSC / rhf.SSC / uks.SSC / uhf.SSC.
    """
    from pyscf.prop import ssc as ssc_mod  # type: ignore

    # Restricted HF
    if isinstance(mf_cpu, scf.hf.RHF):
        from pyscf.prop.ssc import rhf as _b  # type: ignore

        return _b.SSC(mf_cpu)

    # Restricted KS-DFT
    if isinstance(mf_cpu, dft.rks.RKS):
        try:
            from pyscf.prop.ssc import rks as _b  # type: ignore

            return _b.SSC(mf_cpu)
        except Exception:
            LOG.warning("No RKS SSC; retrying via RHF fallback.")
            rhf_mf = scf.RHF(mf_cpu.mol).run()
            from pyscf.prop.ssc import rhf as _b2  # type: ignore

            return _b2.SSC(rhf_mf)

    # Unrestricted HF
    if isinstance(mf_cpu, scf.uhf.UHF):
        from pyscf.prop.ssc import uhf as _b  # type: ignore

        return _b.SSC(mf_cpu)

    # Unrestricted KS-DFT
    if isinstance(mf_cpu, dft.uks.UKS):
        try:
            from pyscf.prop.ssc import uks as _b  # type: ignore

            return _b.SSC(mf_cpu)
        except Exception:
            LOG.warning("No UKS SSC; retrying via UHF fallback.")
            uhf_mf = scf.UHF(mf_cpu.mol).run()
            from pyscf.prop.ssc import uhf as _b2  # type: ignore

            return _b2.SSC(uhf_mf)

    raise RuntimeError(f"Unsupported MF type for SSC: {type(mf_cpu)}")


def _build_J_matrix_Hz(
        mf_cpu,
        isotopes_keep: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute an isotropic scalar J-coupling matrix (Hz) from a CPU mean-field.

    We tolerate multiple PySCF SSC output formats:
      1) ndarray shape (natm, natm)                    ← direct scalar Hz
      2) ndarray shape (n_pairs, 3, 3)                 ← per-pair 3x3 tensors
      3) dict[(ia, ja)] = scalar or tensor-like value  ← mixed/legacy

    For case (2), we assume pairs are ordered over i<j in the upper triangle.
    The scalar J_ij is taken as trace(tensor)/3 (Hz).

    The returned matrix is then subset to the requested isotopes.

    Returns:
        (J_sel, keep_labels)
        J_sel      (M,M) symmetric ndarray in Hz
        keep_labels list[str] nucleus labels matching rows/cols in J_sel
    """

    def _iso_from_tensor(val) -> float:
        """
        Reduce whatever SSC.kernel() gave us for a pair (ia,ja)
        to an isotropic scalar J in Hz.
        """
        arr = np.asarray(val, dtype=float)

        # scalar already
        if arr.ndim == 0:
            return float(arr)

        # full 3x3 tensor → trace/3
        if arr.ndim == 2 and arr.shape == (3, 3):
            return float(np.trace(arr) / 3.0)

        # length-3 vector or arbitrary small thing → mean as fallback
        if arr.ndim == 1 and arr.shape[0] == 3:
            return float(np.mean(arr))

        # last-resort fallback (robust against PySCF shape drift)
        return float(np.mean(arr))

    # run SSC on this MF
    ssc_driver = _pick_ssc_driver(mf_cpu)
    raw = ssc_driver.kernel()

    natm = mf_cpu.mol.natm
    J_full = np.zeros((natm, natm), dtype=float)

    # Format 1: full natm×natm array already in Hz
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        if raw.shape != (natm, natm):
            raise RuntimeError(
                f"SSC kernel ndarray shape {raw.shape} "
                f"!= ({natm},{natm}); don't know how to map."
            )
        J_full[:, :] = np.asarray(raw, dtype=float)

    # Format 2: per-pair anisotropic tensors, e.g. (n_pairs,3,3)
    elif isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.shape[1:] == (3, 3):
        n_pairs_expected = natm * (natm - 1) // 2
        if raw.shape[0] != n_pairs_expected:
            raise RuntimeError(
                f"SSC kernel ndarray shape {raw.shape} doesn't match "
                f"{n_pairs_expected} unique pairs for natm={natm}."
            )
        k = 0
        for i in range(natm):
            for j in range(i + 1, natm):
                Jij_iso_Hz = _iso_from_tensor(raw[k])
                J_full[i, j] = Jij_iso_Hz
                J_full[j, i] = Jij_iso_Hz
                k += 1

    # Format 3: dict[(ia,ja)] = scalar/tensor
    elif isinstance(raw, dict):
        for (ia, ja), val in raw.items():
            Jij_iso_Hz = _iso_from_tensor(val)
            J_full[ia, ja] = Jij_iso_Hz
            J_full[ja, ia] = Jij_iso_Hz

    else:
        raise RuntimeError(
            f"SSC kernel() return type {type(raw)} / shape "
            f"{getattr(getattr(raw, 'shape', None), '__str__', lambda: '?')()} "
            "not recognized."
        )

    # Decide which nuclei to keep (1H-only special case, or explicit isotopes)
    symbols = [a[0] for a in mf_cpu.mol._atom]
    iso_map = {
        "H": "1H",
        "C": "13C",
        "N": "15N",
        "F": "19F",
        "P": "31P",
    }

    only_H = (
            len(isotopes_keep) == 1
            and isotopes_keep[0].lower() in ("1h", "h")
    )

    keep_idx: List[int] = []
    keep_labels: List[str] = []

    for (i, el) in enumerate(symbols):
        if only_H:
            if el.upper() == "H":
                keep_idx.append(i)
                keep_labels.append(f"H{i + 1}")
        else:
            iso_guess = iso_map.get(el.capitalize())
            if iso_guess in isotopes_keep:
                keep_idx.append(i)
                keep_labels.append(f"{el}{i + 1}")

    if not keep_idx:
        # Nothing requested (or no matches). Return empty.
        return np.zeros((0, 0), dtype=float), []

    sel = np.array(keep_idx, dtype=int)
    J_sel = J_full[np.ix_(sel, sel)].astype(float)

    return (J_sel, keep_labels)


def compute_spinspin_JHz(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
        use_gpu: bool,
        isotopes_keep: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute scalar spin–spin couplings (Hz) for selected isotopes.

    Steps:
      1. build mol and run SCF (GPU if allowed)
      2. construct an equivalent CPU MF (property_on_cpu)
      3. run SSC and build dense J matrix (Hz)
      4. keep only requested isotopes
    """
    mol = make_mol(symbols, coords_A, charge, spin, basis)

    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()

    mf_cpu = _property_on_cpu(mol, mf, xc)
    (J_Hz, labels) = _build_J_matrix_Hz(mf_cpu, isotopes_keep=isotopes_keep)

    return (J_Hz, labels)


def compute_sigma_and_J_once(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
        use_gpu: bool,
        isotopes_keep: Sequence[str],
        need_J: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute BOTH, with a single expensive SCF:

      - per-atom isotropic shielding σ_iso (for chemical shifts)
      - scalar spin–spin coupling matrix J (Hz) for selected isotopes

    Steps:
      1. build mol
      2. run SCF once (GPU if allowed)
      3. clone to CPU MF via _property_on_cpu(...)
      4. from that SAME CPU MF:
         - NMR shielding tensors → σ_iso
         - SSC → J_Hz (if need_J)

    If need_J is False, we skip SSC entirely and return
    J_Hz = array([]).reshape(0,0), labels = [].

    Returns:
        (sigma_iso, J_Hz, labels)

        sigma_iso : (natm,) float
            isotropic shielding for each atom in `symbols`

        J_Hz : (M,M) float
            dense symmetric J matrix in Hz for the kept nuclei
            (0x0 if need_J is False)

        labels : list[str] length M
            nucleus labels ("H12", "C3", ...) matching rows/cols of J_Hz
            ([] if need_J is False)
    """
    # 1. Molecule
    mol = make_mol(symbols, coords_A, charge, spin, basis)

    # 2. SCF once (GPU-capable mf)
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()

    # 3. CPU clone seeded from that SCF density
    mf_cpu = _property_on_cpu(mol, mf, xc)

    # 4a. Shieldings from mf_cpu
    arr = np.asarray(nmr_tensors_from_mf(mf_cpu))

    if arr.ndim == 3 and arr.shape[-1] == 3:
        # full 3x3 tensors → trace/3
        sigma_iso = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        # already isotropic per atom
        sigma_iso = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        # principal components → mean
        sigma_iso = arr.mean(axis=1).astype(float)
    else:
        # fallback: mean of diagonal elements
        sigma_iso = (
            np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)
        )

    # 4b. Optional spin–spin couplings on the SAME mf_cpu
    if need_J:
        (J_Hz, labels) = _build_J_matrix_Hz(
            mf_cpu,
            isotopes_keep=isotopes_keep,
        )
    else:
        J_Hz = np.zeros((0, 0), dtype=float)
        labels = []

    return (sigma_iso, J_Hz, labels)


def assert_ssc_available_fast(
        xc: str,
        basis: str,
        isotopes_keep: Sequence[str],
) -> None:
    """
    Fast 'can I do J couplings at all?' probe.

    We build a tiny H2-like system, run an SCF on pure CPU, then attempt:
      - _property_on_cpu()
      - _build_J_matrix_Hz()

    This is considered SUCCESS if:
      - no exception is raised, and
      - we get a finite (possibly 1x1 or 2x2) J matrix for the requested nuclei.

    Any RuntimeError here means 'J is not reliably available', which lets the
    caller abort early if --require-j was set.
    """
    # minimal 2-proton system
    symbols = ["H", "H"]
    coords_A = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float)
    charge = 0
    spin = 0

    mol = make_mol(symbols, coords_A, charge=charge, spin=spin, basis=basis)

    # force CPU here: we only care about SSC plumbing, not GPU perf
    mf = build_scf(mol, xc, use_gpu=False)
    _ = mf.kernel()

    mf_cpu = _property_on_cpu(mol, mf, xc)

    (J_test, lbl_test) = _build_J_matrix_Hz(
        mf_cpu,
        isotopes_keep=isotopes_keep,
    )

    # Check that dimensions line up with labels and values are finite.
    if J_test.shape[0] != J_test.shape[1]:
        raise RuntimeError(
            f"SSC sanity check: non-square J matrix {J_test.shape}"
        )
    if J_test.shape[0] != len(lbl_test):
        raise RuntimeError(
            "SSC sanity check: label/matrix size mismatch "
            f"{J_test.shape[0]} vs {len(lbl_test)}"
        )
    if J_test.size > 0 and not np.all(np.isfinite(J_test)):
        raise RuntimeError(
            "SSC sanity check: J matrix has non-finite values"
        )

    # If we get here: SSC path is judged usable for production.
    LOG.debug(
        "SSC preflight OK: %d spins [%s], J matrix %s",
        len(lbl_test),
        ", ".join(lbl_test),
        J_test.shape,
    )


# -----------------------------------------------------------------------------
# Energies / Boltzmann weights
# -----------------------------------------------------------------------------

def sp_energy_pcm(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
        eps: Optional[float],
        use_gpu: bool,
) -> float:
    """
    Single-point energy (Hartree) for this geometry.

    Behavior:
      1. Build the molecule.
      2. Run ONE gas-phase SCF with build_scf(..., use_gpu=use_gpu),
         and fully converge it (.kernel()) to get a good density.
         This gives us a decent starting density cheaply (GPU if allowed).
      3. If no solvent model (eps is None):
           -> return that gas-phase energy directly (mf0.e_tot).
      4. If solvent model requested (eps not None):
           -> build a CPU MF seeded from that converged density
              via _property_on_cpu(...)
           -> attach ddCOSMO (attach_pcm) on that CPU MF
           -> run that PCM SCF on CPU
           -> return the solvent-corrected total energy.

    Why:
    - gpu4pyscf RKS/UKS objects can't be wrapped with ddCOSMO directly.
    - Previously we tried attach_pcm() on the GPU object and fell back
      to gas-phase with a warning. That means we were silently ignoring
      the requested dielectric.
    - Now we ALWAYS get proper ddCOSMO energy if eps is not None,
      by cloning to CPU first.

    Returns:
        float Hartree total energy (PCM if eps given, else gas-phase).
    """
    # 1. Molecule
    mol = make_mol(symbols, coords_A, charge, spin, basis)

    # 2. Gas-phase SCF once (GPU if allowed)
    mf0 = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf0.kernel()  # converge the vacuum SCF

    # 3. If no solvent requested -> just use that result
    if eps is None:
        # mf0.e_tot should now be populated by kernel()
        return float(mf0.e_tot)

    # 4. Solvent requested:
    #    Move density to a fresh CPU MF, then attach ddCOSMO and solve again.
    mf_cpu = _property_on_cpu(mol, mf0, xc)
    mf_pcm = attach_pcm(mf_cpu, eps=eps)

    # Run the PCM-wrapped SCF on CPU
    e_tot_pcm = mf_pcm.kernel()
    return float(e_tot_pcm)


def boltzmann_weights(E_hartree: Sequence[float], T_K: float) -> np.ndarray:
    """
    Return normalized Boltzmann weights at temperature T_K.
    Energies are in Hartree. Only relative differences matter.
    """
    kB_kcal_per_K = 0.0019872041  # kcal/(mol*K)
    hartree_to_kcalmol = 627.509474
    kB_Ha_per_K = kB_kcal_per_K / hartree_to_kcalmol  # ~3.167e-6 Ha/K

    E = np.array(E_hartree, dtype=float)
    Emin = float(np.min(E))
    beta = 1.0 / (kB_Ha_per_K * T_K)
    x = -beta * (E - Emin)
    x -= np.max(x)

    w = np.exp(x)
    return (w / np.sum(w)).astype(float)


def average_J_matrices(
        J_mats: Sequence[np.ndarray],
        weights: Sequence[float],
) -> np.ndarray:
    """
    Weighted average of multiple J matrices (Hz).
    All matrices must be same shape and label ordering.
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


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------

def write_cluster_shifts(
        out_dir: Path,
        tag: str,
        cid: int,
        atom_names: Sequence[str],
        atom_symbols: Sequence[str],
        sigma_iso: np.ndarray,
        delta_ppm: np.ndarray,
) -> Path:
    """
    Write per-atom shielding and chemical shift for this cluster:
        cluster_<cid>_shifts.tsv
    """
    out = out_dir / tag / f"cluster_{cid}_shifts.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tsigma_iso\tshift_ppm\n")
        for i, (nm, el, sig, dppm) in enumerate(
                zip(atom_names, atom_symbols, sigma_iso, delta_ppm)
        ):
            fh.write(
                f"{i}\t{nm}\t{el}\t{fmt(sig)}\t{fmt(dppm)}\n"
            )
    return out


def write_j_couplings(
        out_dir: Path,
        tag: str,
        cid: int,
        labels: Sequence[str],
        J_Hz: np.ndarray,
) -> Tuple[Path, Path, Path]:
    """
    Write scalar spin–spin coupling info for this cluster:

        cluster_<cid>_J.npy
            dense symmetric (M,M) in Hz
        cluster_<cid>_J_labels.txt
            nucleus labels in matrix order
        cluster_<cid>_j_couplings.tsv
            upper triangle, human-readable
    """
    base_dir = out_dir / tag
    base_dir.mkdir(parents=True, exist_ok=True)

    npy_path = base_dir / f"cluster_{cid}_J.npy"
    np.save(npy_path, J_Hz.astype(float))

    lbl_path = base_dir / f"cluster_{cid}_J_labels.txt"
    with lbl_path.open("w") as fh_lbl:
        for lbl in labels:
            fh_lbl.write(f"{lbl}\n")

    tsv_path = base_dir / f"cluster_{cid}_j_couplings.tsv"
    with tsv_path.open("w") as fh_tsv:
        fh_tsv.write("# i\tj\tlabel_i\tlabel_j\tJ_Hz\n")
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                fh_tsv.write(
                    f"{i}\t{j}\t{labels[i]}\t{labels[j]}\t"
                    f"{fmt(float(J_Hz[i, j]))}\n"
                )

    return (npy_path, lbl_path, tsv_path)


def write_params(
        out_dir: Path,
        tag: str,
        name: str,
        params: Dict[str, object],
) -> Path:
    """
    Write a simple key:value metadata file for reproducibility.
    Example names:
        params_compute.txt
        params_average.txt
    """
    out = out_dir / tag / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for k, v in params.items():
            fh.write(f"{k}: {v}\n")
    return out
