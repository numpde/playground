#!/usr/bin/env python3
# f_predict_shifts_core.py
#
# Core utilities for NMR shift / J coupling / Boltzmann weighting pipeline.
#
# Revised 2025-11-02:
#   - proper SSC driver selection (pyscf.prop.ssc.rks / rhf / uks / uhf),
#   - J matrix construction + file writers,
#   - fast capability check to fail early if SSC/PCM props are missing,
#   - clearer ddCOSMO fallback semantics.
#
# This module is imported by:
#   - f_predict_shifts_compute.py   (per-cluster quantum calcs)
#   - f_predict_shifts_average.py   (Boltzmann averaging / fast exchange)

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
from pyscf import gto, dft, scf

# Silence PySCF "under testing" noise, keep real errors visible
warnings.filterwarnings("ignore", message=r"Module .* is under testing", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Module .* is not fully tested", category=UserWarning)

LOG = logging.getLogger("nmrshifts.core")

# ---- Optional GPU backend -------------------------------------------------
try:
    from gpu4pyscf import dft as g4dft  # type: ignore

    GPU4PYSCF_AVAILABLE = True
except Exception:
    GPU4PYSCF_AVAILABLE = False

# ---- Optional Berny (geometry optimizer) ----------------------------------
try:
    import berny  # type: ignore

    BERNY_AVAILABLE = True
except Exception:
    BERNY_AVAILABLE = False

# ---- Paths / defaults -----------------------------------------------------
OUT_DIR = Path("f_predict_shifts")
CLUSTERS_DIR = Path("e_cluster")
DFT_XC_DEFAULT = "b3lyp"
BASIS_DEFAULT = "def2-tzvp"
SCF_MAXCYC = 200
SCF_CONV_TOL = 1e-9
GRAD_CONV_TOL = 3e-4  # not currently used, but kept for possible geom opt

# ---- Solvent map (eps ~298 K). Override with --eps ------------------------
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


# -----------------------------------------------------------------------------
# Logging / convenience
# -----------------------------------------------------------------------------

@dataclass
class ClusterRow:
    cid: int
    fraction: float  # MD or clustering population fraction
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
    """Heuristic net charge / spin multiplicity guess from tag name."""
    t = tag.lower()
    if "deprot" in t or "anion" in t:
        return (-1, 0)
    if "cation" in t or "prot" in t:
        return (+1, 0)
    return (0, 0)


def guess_rep_path(tag: str, cid: int) -> Path:
    """Try a few filename patterns to locate the cluster representative PDB."""
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
    # final fallback (even if it doesn't exist yet)
    return p


def load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    """
    Load cluster summary for `tag`.

    Supports two formats:

    A) A TSV with named headers (cid/fraction/rep_pdb/etc.). We try to guess
       which columns are which by fuzzy name matching.

    B) The "commented TSV" format written by e_cluster.py:
       #cluster_id  count  fraction  medoid_frame_idx  rep_pdb_path
       0            123    0.2049    157               foo_cluster_0_rep.pdb
       ...

    Returns
    -------
    rows : list[ClusterRow]
    """
    # --- attempt A: parse as dataframe with headers -------------------------
    try:
        df_try = pd.read_csv(path, sep=None, engine="python", comment="#").copy()
        cols = {c.lower(): c for c in df_try.columns}

        cid_col = None
        for cand in ("cid", "cluster", "cluster_id", "clusteridx",
                     "cluster_idx", "id"):
            if cand in cols:
                cid_col = cols[cand]
                break

        if cid_col is not None:
            # we think this has headers
            frac_col = None
            for cand in ("fraction", "pop", "population", "weight",
                         "boltz", "md_fraction", "mdfrac", "md_weight"):
                if cand in cols:
                    frac_col = cols[cand]
                    break

            rep_col = None
            for cand in ("rep", "rep_pdb", "rep_path", "pdb", "pdb_path",
                         "representative"):
                if cand in cols:
                    rep_col = cols[cand]
                    break

            cidv = pd.to_numeric(df_try[cid_col], errors="raise")

            if frac_col is not None:
                frv = pd.to_numeric(df_try[frac_col], errors="coerce")
            else:
                frv = None

            if rep_col is not None:
                rpv = df_try[rep_col].astype(str)
            else:
                rpv = None

            rows: List[ClusterRow] = []
            for i in range(len(df_try)):
                cid_i = int(cidv.iloc[i])

                # fraction
                if frv is not None and pd.notna(frv.iloc[i]):
                    frac_i = float(frv.iloc[i])
                else:
                    frac_i = float("nan")

                # representative pdb path
                if rpv is not None and rpv.iloc[i].strip():
                    rp_raw = Path(rpv.iloc[i].strip())
                    rep_path = (
                        rp_raw if rp_raw.is_absolute()
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

            # If any fractions are NaN, fill them uniformly
            fracs = np.array([r.fraction for r in rows], dtype=float)
            if not np.all(np.isfinite(fracs)):
                n = len(rows)
                for r in rows:
                    r.fraction = 1.0 / n

            return rows

    except Exception:
        # fall through to attempt B
        pass

    # --- attempt B: manual parse of "commented TSV" -------------------------
    rows: List[ClusterRow] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # columns: cluster_id  count  fraction  medoid_frame_idx  rep_pdb_path
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

    # fill uniform fractions if needed
    fracs = np.array([r.fraction for r in rows], dtype=float)
    if not np.all(np.isfinite(fracs)):
        n = len(rows)
        for r in rows:
            r.fraction = 1.0 / n

    return rows


def read_pdb_atoms(pdb: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Minimal PDB reader: return (atom_names, atom_symbols, coords_A[n,3]).
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
# SCF / property helpers
# -----------------------------------------------------------------------------

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
    """
    Create an RKS/UKS object, preferring gpu4pyscf if requested+available.
    We only *build* here; caller must run .kernel().
    """
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

    mf.xc = xc
    mf.max_cycle = SCF_MAXCYC
    mf.conv_tol = SCF_CONV_TOL
    return mf


def _property_on_cpu(mol, gpu_mf, xc: str):
    """
    Build a *CPU* MF using density from possibly-GPU mf, and run .kernel().
    This lets us call PySCF property drivers (NMR, SSC, PCM) that gpu4pyscf
    doesn't implement.
    """
    mf_cpu = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf_cpu.xc = xc
    mf_cpu.max_cycle = SCF_MAXCYC
    mf_cpu.conv_tol = SCF_CONV_TOL

    # try to seed with GPU density
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
    Try to wrap an MF object with ddCOSMO. If unsupported, return the original
    MF and warn. NOTE: gpu4pyscf objects generally don't support solvent
    attachments; caller should usually bounce to CPU first for PCM.
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
    Call PySCF's NMR shielding driver on mf and return raw shielding tensors.

    We try modern API (pyscf.prop.nmr.NMR) first, fall back to mf.NMR(),
    then versioned backends (rks/uks/rhf/uhf). Raise RuntimeError on failure.
    """
    # new consolidated API
    try:
        from pyscf.prop.nmr import NMR  # type: ignore
        return NMR(mf).kernel()
    except Exception:
        pass

    # old attached method
    if hasattr(mf, "NMR"):
        try:
            return mf.NMR().kernel()
        except Exception:
            pass

    # backend-specific fallback
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

def compute_sigma_iso(symbols,
                      coords_A,
                      charge,
                      spin,
                      xc,
                      basis,
                      use_gpu: bool):
    """
    Return per-nucleus isotropic shielding sigma_iso[i] (dimensionless).

    We do:
      1. build mol
      2. SCF on GPU if allowed
      3. run shielding property; if gpu4pyscf can't do it,
         bounce density to CPU and retry.
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

    # arr can be [natm,3,3] or [natm] or [natm,3] depending on PySCF version
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # full tensor per atom
        sigma_iso = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        sigma_iso = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        sigma_iso = arr.mean(axis=1).astype(float)
    else:
        sigma_iso = np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)

    return sigma_iso


def sigma_to_delta(symbols: Sequence[str],
                   sigma_iso: np.ndarray,
                   ref_sigma: Dict[str, float]) -> np.ndarray:
    """
    Convert sigma_iso to chemical shifts δ (ppm) via δ = σ_ref - σ.
    We only fill for nuclei we have refs for (H,C by default).
    """
    out = np.full_like(sigma_iso, np.nan, dtype=float)
    for i, el in enumerate(symbols):
        elU = el.upper()
        if elU in ("H", "C"):
            out[i] = ref_sigma[elU] - sigma_iso[i]
    return out


def tms_geometry() -> Tuple[List[str], np.ndarray]:
    """
    Build an approximate TMS geometry (Si(CH3)4) for reference shielding,
    with decent Si–C / C–H distances and tetrahedral directions.
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

        # build a crude 109° CH3 fan
        (ax, ay, az) = (-ux, -uy, -uz)
        if abs(ax) < 0.9:
            (px, py, pz) = (1.0, 0.0, 0.0)
        else:
            (px, py, pz) = (0.0, 1.0, 0.0)

        # orthonormalize px -> vx, and vx×ax -> wx
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

        for ang in (0.0, 2.0943951023931953, 4.1887902047863905):  # 0,120,240 deg
            hx = cx + R_CH * (0.6 * ax + 0.8 * (math.cos(ang) * vx + math.sin(ang) * wx))
            hy = cy + R_CH * (0.6 * ay + 0.8 * (math.cos(ang) * vy + math.sin(ang) * wy))
            hz = cz + R_CH * (0.6 * az + 0.8 * (math.cos(ang) * vz + math.sin(ang) * wz))
            add("H", hx, hy, hz)

    return syms, np.array(coords, float)


@lru_cache(maxsize=1)
def tms_ref_sigma(xc: str, basis: str) -> Dict[str, float]:
    """
    Compute and cache σ_ref for TMS at the chosen (xc,basis).

    We run RKS on CPU (no GPU needed), compute NMR tensors,
    then average all equivalent H and C.
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

    Hvals = [sigma[i] for (i, s) in enumerate(syms) if s.upper() == "H"]
    Cvals = [sigma[i] for (i, s) in enumerate(syms) if s.upper() == "C"]
    H = float(np.mean(Hvals))
    C = float(np.mean(Cvals))
    return {"H": H, "C": C}


# -----------------------------------------------------------------------------
# Scalar spin-spin couplings (J in Hz)
# -----------------------------------------------------------------------------

def _pick_ssc_driver(mf_cpu):
    """
    Return an SSC driver bound to mf_cpu using the correct backend.

    PySCF exposes SSC as separate classes in pyscf.prop.ssc.rhf / rks / uhf / uks
    instead of a single generic entry. We'll try best match.
    """
    from pyscf.prop import ssc as ssc_mod  # type: ignore

    # Restricted HF?
    if isinstance(mf_cpu, scf.hf.RHF):
        from pyscf.prop.ssc import rhf as _b  # type: ignore
        return _b.SSC(mf_cpu)

    # Restricted KS-DFT?
    if isinstance(mf_cpu, dft.rks.RKS):
        try:
            from pyscf.prop.ssc import rks as _b  # type: ignore
            return _b.SSC(mf_cpu)
        except Exception:
            # fallback: degrade to RHF driver
            LOG.warning("No RKS SSC found; retrying via RHF fallback.")
            rhf_mf = scf.RHF(mf_cpu.mol).run()
            from pyscf.prop.ssc import rhf as _b2  # type: ignore
            return _b2.SSC(rhf_mf)

    # Unrestricted HF?
    if isinstance(mf_cpu, scf.uhf.UHF):
        from pyscf.prop.ssc import uhf as _b  # type: ignore
        return _b.SSC(mf_cpu)

    # Unrestricted KS-DFT?
    if isinstance(mf_cpu, dft.uks.UKS):
        try:
            from pyscf.prop.ssc import uks as _b  # type: ignore
            return _b.SSC(mf_cpu)
        except Exception:
            LOG.warning("No UKS SSC found; retrying via UHF fallback.")
            uhf_mf = scf.UHF(mf_cpu.mol).run()
            from pyscf.prop.ssc import uhf as _b2  # type: ignore
            return _b2.SSC(uhf_mf)

    raise RuntimeError(f"Unsupported MF type for SSC: {type(mf_cpu)}")


def _build_J_matrix_Hz(mf_cpu,
                       isotopes_keep: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Run SSC kernel on mf_cpu, slice to kept isotopes, and return (J_Hz, labels).

    We handle two PySCF patterns:
      - ssc_driver.kernel() -> ndarray [natm,natm] (Hz)
      - ssc_driver.kernel() -> dict[(ia,ja)] = tensor or scalar

    We also generate per-spin labels like "H1","H2",... in that order.
    """
    ssc_driver = _pick_ssc_driver(mf_cpu)
    raw = ssc_driver.kernel()

    # Which nuclei do we even want?
    # Map element -> default NMR-active isotope tag
    iso_map = {
        "H": "1H",
        "C": "13C",
        "N": "15N",
        "F": "19F",
        "P": "31P",
    }

    symbols = [a[0] for a in mf_cpu.mol._atom]
    keep_idx: List[int] = []
    keep_labels: List[str] = []

    # Special case: user passes only ["1H"] -> keep only hydrogens
    only_H = (
            len(isotopes_keep) == 1
            and isotopes_keep[0] in ("1H", "H", "1h", "h")
    )

    for (i, el) in enumerate(symbols):
        if only_H:
            if el.upper() == "H":
                keep_idx.append(i)
                keep_labels.append(f"H{i + 1}")
            continue

        iso_guess = iso_map.get(el.capitalize())
        if iso_guess in isotopes_keep:
            keep_idx.append(i)
            keep_labels.append(f"{el}{i + 1}")

    if not keep_idx:
        return np.zeros((0, 0), dtype=float), []

    natm = mf_cpu.mol.natm

    # Case A: ndarray [natm,natm] already in Hz
    if isinstance(raw, np.ndarray):
        if raw.shape[0] != natm or raw.shape[1] != natm:
            raise RuntimeError(
                f"SSC kernel ndarray shape {raw.shape} != ({natm},{natm})"
            )
        J_full = np.asarray(raw, dtype=float)

    # Case B: dict[(ia,ja)] -> coupling info; need to assemble
    elif isinstance(raw, dict):
        J_full = np.zeros((natm, natm), dtype=float)
        for (ia, ja), val in raw.items():
            # val might be scalar Hz, or a 3x3 tensor (au); take isotropic
            if np.isscalar(val):
                Jij_iso_Hz = float(val)
            else:
                arr = np.asarray(val, dtype=float)
                # if it's a 3x3 tensor, take isotropic piece (trace/3)
                if arr.ndim == 2 and arr.shape == (3, 3):
                    Jij_iso_Hz = float(np.trace(arr) / 3.0)
                else:
                    # last resort: average all components
                    Jij_iso_Hz = float(np.mean(arr))
            J_full[ia, ja] = Jij_iso_Hz
            J_full[ja, ia] = Jij_iso_Hz
    else:
        raise RuntimeError(
            f"Unsupported SSC kernel() return type: {type(raw)}"
        )

    # Slice down to kept nuclei
    keep_idx_arr = np.array(keep_idx, dtype=int)
    J_sel = J_full[np.ix_(keep_idx_arr, keep_idx_arr)].astype(float)

    return (J_sel, keep_labels)


def compute_spinspin_JHz(symbols,
                         coords_A,
                         charge,
                         spin,
                         xc,
                         basis,
                         use_gpu: bool,
                         isotopes_keep: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute scalar spin-spin couplings (Hz) for selected isotopes.

    Steps:
      1. build mol, run SCF (GPU if allowed)
      2. bounce MF to CPU (mf_cpu) so we can call PySCF SSC driver
      3. run SSC, build dense symmetric J matrix in Hz
      4. keep only desired isotopes, return (J_Hz, labels)

    Raises RuntimeError if SSC is not available.
    """
    mol = make_mol(symbols, coords_A, charge, spin, basis)

    # SCF on GPU (fast) or CPU
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()

    # bounce to CPU for property
    mf_cpu = _property_on_cpu(mol, mf, xc)

    # build & slice J matrix
    (J_Hz, labels) = _build_J_matrix_Hz(mf_cpu, isotopes_keep=isotopes_keep)

    return (J_Hz, labels)


def assert_ssc_available_fast(xc: str,
                              basis: str,
                              isotopes_keep: Sequence[str]) -> None:
    """
    Cheap pre-flight check: try to compute a tiny SSC on a trivial diatomic,
    *on CPU only*, so we can fail fast before doing hour-long DFT on
    strychnine.

    Call this once at the start of f_predict_shifts_compute.py
    (before looping over clusters). If this raises, abort immediately.
    """
    # minimal 2-atom test system (H2 ~0.74 Å)
    symbols = ["H", "H"]
    coords_A = np.array([[0.0, 0.0, 0.0],
                         [0.74, 0.0, 0.0]], dtype=float)

    mol = make_mol(symbols, coords_A, charge=0, spin=0, basis=basis)
    mf = build_scf(mol, xc, use_gpu=False)
    _ = mf.kernel()
    mf_cpu = _property_on_cpu(mol, mf, xc)

    try:
        _ = _build_J_matrix_Hz(mf_cpu, isotopes_keep=isotopes_keep)
    except Exception as e:
        raise RuntimeError(
            f"SSC / spin-spin coupling appears unavailable in this PySCF build: {e}"
        )


# -----------------------------------------------------------------------------
# Energies / Boltzmann weights
# -----------------------------------------------------------------------------

def sp_energy_pcm(symbols,
                  coords_A,
                  charge,
                  spin,
                  xc,
                  basis,
                  eps: Optional[float],
                  use_gpu: bool) -> float:
    """
    Single-point electronic energy in Hartree for this geometry.

    We *attempt* ddCOSMO with dielectric eps (solvent-like). gpu4pyscf
    objects generally don't support solvent, so:
        - we run SCF (GPU if allowed),
        - then we call attach_pcm(...) on that MF,
        - if attach_pcm fails (ddCOSMO not available on that object),
          we warn and give gas-phase energy instead.

    We only use these energies for RELATIVE Boltzmann weights, not absolute
    solvation free energies.
    """
    mol = make_mol(symbols, coords_A, charge, spin, basis)
    mf0 = build_scf(mol, xc, use_gpu=use_gpu)
    mf_pcm = attach_pcm(mf0, eps=eps)
    e_tot = mf_pcm.kernel()
    return float(e_tot)


def boltzmann_weights(E_hartree: Sequence[float], T_K: float) -> np.ndarray:
    """
    Return normalized Boltzmann weights w[i] at temperature T_K
    from energies E[i] in Hartree.

    We subtract min(E) for numerical stability, convert Hartree to kcal/mol,
    and normalize exp(-ΔE/kT).
    """
    kB_kcal_per_K = 0.0019872041  # Boltzmann in kcal/(mol*K)
    hartree_to_kcalmol = 627.509474
    kB_Ha_per_K = kB_kcal_per_K / hartree_to_kcalmol  # ~3.167e-6 Ha/K

    E = np.array(E_hartree, dtype=float)
    Emin = float(np.min(E))
    beta = 1.0 / (kB_Ha_per_K * T_K)
    x = -beta * (E - Emin)
    x -= np.max(x)
    w = np.exp(x)
    return (w / np.sum(w)).astype(float)


def average_J_matrices(J_mats: Sequence[np.ndarray],
                       weights: Sequence[float]) -> np.ndarray:
    """
    Weighted average of multiple J matrices (Hz).

    All J_mats[k] must be the same shape (M,M) and correspond to the same
    nucleus ordering / labels. weights[k] can be any non-negative numbers
    (Boltzmann weights, MD fractions, ...). We normalize them internally.
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

def write_cluster_shifts(out_dir: Path,
                         tag: str,
                         cid: int,
                         atom_names,
                         atom_symbols,
                         sigma_iso,
                         delta_ppm) -> Path:
    """
    Write per-atom shielding and chemical shift (ppm) for this cluster:
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


def write_j_couplings(out_dir: Path,
                      tag: str,
                      cid: int,
                      labels: Sequence[str],
                      J_Hz: np.ndarray) -> Tuple[Path, Path, Path]:
    """
    Persist J-coupling info for this cluster:

      cluster_<cid>_J.npy            (M,M dense symmetric, Hz)
      cluster_<cid>_J_labels.txt     (1 label per row/col)
      cluster_<cid>_j_couplings.tsv  (upper triangle, tidy text)

    The TSV columns:
        i   j   label_i   label_j   J_Hz
    """
    base_dir = out_dir / tag
    base_dir.mkdir(parents=True, exist_ok=True)

    # Dense matrix (machine use)
    npy_path = base_dir / f"cluster_{cid}_J.npy"
    np.save(npy_path, J_Hz)

    # Labels
    lbl_path = base_dir / f"cluster_{cid}_J_labels.txt"
    with lbl_path.open("w") as fh_lbl:
        for lbl in labels:
            fh_lbl.write(f"{lbl}\n")

    # Upper triangle TSV
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


def write_params(out_dir: Path,
                 tag: str,
                 name: str,
                 params: Dict[str, object]) -> Path:
    """
    Write a simple key: value metadata file for reproducibility.
    Example:
        params_compute.txt
        params_average.txt
    """
    out = out_dir / tag / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for k, v in params.items():
            fh.write(f"{k}: {v}\n")
    return out
