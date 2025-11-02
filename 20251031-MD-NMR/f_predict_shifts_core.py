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
    Build an idealized Td TMS (Si(CH3)4) geometry in Cartesian Å.

    - Si at the origin.
    - 4 carbons along perfect tetrahedral directions (±1,±1,±1 normalized)
      at distance R_SiC.
    - For each carbon, place 3 hydrogens in perfect tetrahedral geometry
      around that carbon, assuming Si is the 4th substituent.

    The CH3 geometry is constructed analytically:
    - Let the C–Si bond define the local +z axis (pointing from C to Si).
    - In a perfect sp3 tetrahedron, the angle between any two bonds is
      arccos(-1/3) ≈ 109.47°. That means each C–H bond direction has
      cos(theta) = -1/3 relative to +z, and radial component in the
      perpendicular plane of magnitude sin(theta) = 2*sqrt(2)/3.

    By generating the 3 hydrogens at azimuths 0°, 120°, 240° in that
    local frame, and then rotating that frame so +z maps onto the actual
    C→Si direction, all four methyl groups are strictly equivalent.
    """

    import numpy as np
    import math

    syms: List[str] = []
    coords: List[np.ndarray] = []

    # Distances (Å). These do not need to be extremely accurate;
    # what's critical is *symmetry*, not exact bond length.
    R_SiC = 1.86  # Si–C bond length guess
    R_CH = 1.09  # C–H bond length guess

    # Put Si at origin
    syms.append("Si")
    coords.append(np.zeros(3, dtype=float))

    # Ideal tetrahedral directions for Si–C bonds
    dirs = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]  # normalize each

    # Precompute local tetrahedral CH3 geometry in a canonical frame:
    # local frame: C at origin, Si on +z, hydrogens at 109.47° from +z
    cos_theta = -1.0 / 3.0
    sin_theta = (2.0 * math.sqrt(2.0)) / 3.0  # = sqrt(1 - cos^2)

    # unit vectors for the 3 H positions in the canonical frame
    # azimuth 0°, 120°, 240°
    def canonical_CH_dirs():
        out = []
        for phi_deg in (0.0, 120.0, 240.0):
            phi = math.radians(phi_deg)
            # vector in canonical coords (C at 0, Si along +z)
            vx = sin_theta * math.cos(phi)
            vy = sin_theta * math.sin(phi)
            vz = cos_theta
            out.append(np.array([vx, vy, vz], dtype=float))
        return out

    CH_local_dirs = canonical_CH_dirs()

    for u in dirs:
        # Carbon position relative to Si
        c_pos = R_SiC * u
        syms.append("C")
        coords.append(c_pos)

        # Build a local frame for this methyl:
        # We want the local +z axis to point from C toward Si.
        # axis = (Si - C) normalized = (-u) because Si is at origin.
        axis = (-u).copy()
        axis /= np.linalg.norm(axis)

        # Pick a helper vector not parallel to axis for frame construction
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(helper, axis)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0], dtype=float)

        # Orthonormal basis: e3 = axis, e1 ⟂ axis, e2 = e3 × e1
        e3 = axis
        e1 = np.cross(e3, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e3, e1)

        # Now place the 3 hydrogens
        for d_loc in CH_local_dirs:
            # d_loc is expressed in canonical frame where +z is C→Si.
            # Map it into world coords: d_world = d_loc_x * e1 + d_loc_y * e2 + d_loc_z * e3
            d_world = d_loc[0] * e1 + d_loc[1] * e2 + d_loc[2] * e3
            h_pos = c_pos + R_CH * d_world
            syms.append("H")
            coords.append(h_pos)

    return (syms, np.array(coords, dtype=float))


def _optimize_geometry_cpu(
        syms: Sequence[str],
        coords_A_init: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
) -> np.ndarray:
    """
    Attempt to return a Berny-relaxed Cartesian geometry (Å).

    This is ONLY called when the caller explicitly allows TMS optimization.

    Policy:
      - Try Berny geometry optimization at the requested xc/basis.
        Handle all observed PySCF return conventions:
          * mol
          * (converged_flag, mol)
          * (mol, extra_info)

      - If anything goes wrong, fall back to the symmetric analytic TMS guess
        (coords_A_init), which is internally self-consistent and gives
        δ(TMS) ≈ 0 ppm.

    We do not raise on Berny failure. We warn and degrade gracefully.

    Returns
    -------
    coords_opt : np.ndarray, shape (natm, 3)
        Final Cartesian coordinates in Å.
    """
    mol0 = make_mol(syms, coords_A_init, charge, spin, basis)

    # plain CPU SCF first, to have a sane starting wavefunction
    mf0 = build_scf(mol0, xc, use_gpu=False)
    _ = mf0.kernel()

    if BERNY_AVAILABLE:
        try:
            from pyscf.geomopt.berny_solver import kernel as berny_kernel  # type: ignore

            berny_ret = berny_kernel(mf0, assert_convergence=True)

            # Unwrap berny_ret into an actual Mole-like object with .atom_coords(...)
            mol_candidate = None

            if hasattr(berny_ret, "atom_coords"):
                # classic API: kernel(...) -> mol
                mol_candidate = berny_ret

            elif isinstance(berny_ret, tuple):
                # possibilities:
                #   (converged_flag, mol)
                #   (mol, loginfo)
                # We'll scan both elements and pick the one that looks like a Mole.
                for item in berny_ret:
                    if hasattr(item, "atom_coords"):
                        mol_candidate = item
                        break

            if mol_candidate is None:
                raise RuntimeError(
                    f"Could not extract optimized Mole from berny_ret={type(berny_ret)}"
                )

            coords_opt = np.asarray(
                mol_candidate.atom_coords(unit="Angstrom"),
                dtype=float,
            )
            return coords_opt

        except Exception as e:
            # Berny ran or partially ran but we couldn't parse it
            LOG.warning(
                "Berny optimization failed (%r); "
                "falling back to symmetric guess.",
                e,
            )

    # Fallback: keep the symmetric analytic Td guess unchanged
    LOG.warning(
        "Berny geometry optimization not available; "
        "using unoptimized symmetric TMS guess for reference."
    )
    return np.asarray(coords_A_init, dtype=float)


@lru_cache(maxsize=16)
def tms_geometry(xc: str, basis: str, do_opt: bool) -> Tuple[List[str], np.ndarray]:
    """
    Return a TMS geometry for (xc,basis) in Å, with optional Berny relaxation.

    Parameters
    ----------
    xc : str
        DFT functional.
    basis : str
        Basis set.
    do_opt : bool
        If True:
            - Build an analytic symmetric Td Si(CH3)4 guess.
            - Try Berny geometry optimization with the given xc/basis.
            - On failure, fall back to the symmetric guess and WARN.
        If False (default in the pipeline):
            - Build the symmetric Td guess ONLY.
            - Do NOT call Berny at all.

    Notes
    -----
    We cache on (xc,basis,do_opt) so the cost is paid once.

    Returns
    -------
    (syms, coords_A_opt)
        syms        list[str]   atomic symbols
        coords_A_opt np.ndarray (natm,3) Å
    """
    (syms0, coords0_A) = _tms_guess_geometry()

    if do_opt:
        coords_out_A = _optimize_geometry_cpu(
            syms0,
            coords0_A,
            charge=0,
            spin=0,
            xc=xc,
            basis=basis,
        )
    else:
        # Fast path: skip Berny entirely.
        coords_out_A = np.asarray(coords0_A, dtype=float)

    return (syms0, coords_out_A)


@lru_cache(maxsize=16)
def tms_ref_sigma(xc: str, basis: str, do_opt: bool) -> Dict[str, float]:
    """
    Compute and cache the reference isotropic shieldings σ_ref for TMS at
    (xc,basis), as {'H': <σ_H>, 'C': <σ_C>}.

    This is what defines δ(ppm) via δ = σ_ref - σ.

    Behavior depends on do_opt:

    - do_opt == False (default in compute):
        Use the symmetric analytic Td TMS (no Berny). Skip any geometry
        optimization entirely. This keeps the run cheap and deterministic.

    - do_opt == True (--tms-opt flag):
        Attempt Berny geometry optimization of TMS at (xc,basis).
        If Berny fails or is unavailable, fall back to the symmetric Td guess
        and WARN.

    After we get the geometry:
        1. Run the standard single-shot pipeline on CPU:
            compute_sigma_J_and_energy_once(...)
            with need_J=False, eps=None.
        2. Extract all 1H σ_iso → average → σ_ref['H'].
        3. Extract all 13C σ_iso → average → σ_ref['C'].

    Returns
    -------
    ref_sigma : dict
        {'H': float, 'C': float}
    """
    # 1. Get TMS geometry per requested policy
    (syms, coords_A) = tms_geometry(xc, basis, do_opt)

    # 2. Single-shot SCF → σ_iso using the unified path.
    (
        sigma_iso,
        _J_unused,
        _lbl_unused,
        _e_unused,
    ) = compute_sigma_J_and_energy_once(
        symbols=syms,
        coords_A=coords_A,
        charge=0,
        spin=0,
        xc=xc,
        basis=basis,
        use_gpu=False,  # CPU is fine for this tiny system
        isotopes_keep=["1H", "13C"],  # unused when need_J=False
        need_J=False,  # skip SSC entirely
        eps=None,  # gas-phase energy only
    )

    # 3. Element-wise averages to define σ_ref
    Hvals = [sigma_iso[i] for (i, s) in enumerate(syms) if s.upper() == "H"]
    Cvals = [sigma_iso[i] for (i, s) in enumerate(syms) if s.upper() == "C"]

    if not Hvals or not Cvals:
        raise RuntimeError(
            "tms_ref_sigma(): TMS shielding missing H or C after reference calc"
        )

    ref_H = float(np.mean(Hvals))
    ref_C = float(np.mean(Cvals))

    return {"H": ref_H, "C": ref_C}


# -----------------------------------------------------------------------------
# Scalar spin–spin couplings (J in Hz)
# -----------------------------------------------------------------------------

def _pick_ssc_driver(mf_cpu):
    """
    Return a PySCF spin–spin coupling driver bound to ``mf_cpu`` that can
    actually report physical J couplings in Hz.

    Strategy:
    - Prefer the high-level ``SpinSpinCoupling`` class (it computes and
      prints "Spin-spin coupling constant J (Hz)").
    - Fall back to the lower-level ``SSC`` class if ``SpinSpinCoupling``
      is missing for that backend.

    We branch on the MF type (RHF/RKS/UHF/UKS) so PySCF picks the right
    response theory.

    This replaces the older version that always returned ``SSC``.
    That older path only exposed the reduced K tensors and led to
    ~0 Hz downstream.
    """
    # imports inside the function so module import doesn't hard-require PySCF at import time
    from pyscf import scf, dft
    from pyscf.prop import ssc as ssc_mod

    def _backend_driver(backend_mod, mf_for_backend):
        """
        Given a backend module like ssc_mod.rks or ssc_mod.rhf and
        the matching mean-field object, try SpinSpinCoupling first,
        then SSC.
        """
        if hasattr(backend_mod, "SpinSpinCoupling"):
            return backend_mod.SpinSpinCoupling(mf_for_backend)
        if hasattr(backend_mod, "SSC"):
            return backend_mod.SSC(mf_for_backend)
        return None

    drv = None

    # Restricted HF
    if isinstance(mf_cpu, scf.hf.RHF):
        if hasattr(ssc_mod, "rhf"):
            drv = _backend_driver(ssc_mod.rhf, mf_cpu)

    # Restricted KS-DFT
    elif isinstance(mf_cpu, dft.rks.RKS):
        if hasattr(ssc_mod, "rks"):
            drv = _backend_driver(ssc_mod.rks, mf_cpu)

        # If RKS backend missing, degrade politely to RHF on the same geometry.
        if drv is None and hasattr(ssc_mod, "rhf"):
            LOG.warning("No RKS SpinSpinCoupling/SSC; retrying via RHF fallback.")
            rhf_mf = scf.RHF(mf_cpu.mol).run()
            drv = _backend_driver(ssc_mod.rhf, rhf_mf)

    # Unrestricted HF
    elif isinstance(mf_cpu, scf.uhf.UHF):
        if hasattr(ssc_mod, "uhf"):
            drv = _backend_driver(ssc_mod.uhf, mf_cpu)

    # Unrestricted KS-DFT
    elif isinstance(mf_cpu, dft.uks.UKS):
        if hasattr(ssc_mod, "uks"):
            drv = _backend_driver(ssc_mod.uks, mf_cpu)

        # If UKS backend missing, degrade to UHF.
        if drv is None and hasattr(ssc_mod, "uhf"):
            LOG.warning("No UKS SpinSpinCoupling/SSC; retrying via UHF fallback.")
            uhf_mf = scf.UHF(mf_cpu.mol).run()
            drv = _backend_driver(ssc_mod.uhf, uhf_mf)

    if drv is None:
        raise RuntimeError(f"Unsupported MF type for SSC: {type(mf_cpu)}")

    return drv


def _build_J_matrix_Hz(mf_cpu, isotopes_keep):
    """
    Compute an *isotropic scalar* J–coupling matrix (Hz) for the requested
    nuclei, from one CPU mean-field.

    What changed:
    - We now run PySCF's SpinSpinCoupling/SSC ONCE, capture its stdout,
      and parse the human-readable table
          "Spin-spin coupling constant J (Hz)"
      which is where PySCF actually prints the physical J_ij in Hz.
      Those are the ~7 Hz vicinal 1H–1H, ~130 Hz 1J_CH, ~400+ Hz 1J_HF
      numbers chemists expect.

    - If parsing that table fails (backend differences etc.), we fall
      back to the old tensor trace/3 heuristic on the raw return, so we
      still won't crash.

    Returns:
        (J_sel, keep_labels)
        J_sel        np.ndarray shape (M,M), symmetric in Hz
        keep_labels  list[str]   pretty nucleus labels corresponding to rows/cols
    """
    import io
    import contextlib
    import numpy as np

    natm = mf_cpu.mol.natm
    symbols = [atom[0] for atom in mf_cpu.mol._atom]  # e.g. ["C","H","H",...]

    # ---- 1. Run SSC once, capture stdout ----
    ssc_driver = _pick_ssc_driver(mf_cpu)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        raw = ssc_driver.kernel()
    ssc_text = buf.getvalue()

    # ---- 2. Parse "Spin-spin coupling constant J (Hz)" table ----
    def _parse_J_table(stdout_text, natm):
        """
        Returns (J_full, ok)

        J_full : (natm,natm) float64, symmetric
        ok     : bool, True if we parsed the real J(Hz) table
        """
        Jmat = np.zeros((natm, natm), dtype=float)
        lines = stdout_text.splitlines()

        # Find the block header
        start_idx = None
        for (li, line) in enumerate(lines):
            if "Spin-spin coupling constant J" in line and "(Hz)" in line:
                # Skip two lines:
                #   header with '#0 #1 #2 ...'
                #   then first data row starts
                start_idx = li + 2
                break

        if start_idx is None:
            return (Jmat, False)

        rows_parsed = 0
        for line in lines[start_idx:]:
            if not line.strip():
                break  # blank line = end of table

            # Expected-ish row format, e.g.
            # "        1 F  413.92070   0.00000"
            parts = line.strip().split()
            # parts[0] -> row atom index (int)
            # parts[1] -> element symbol ("H","C",...)
            # parts[2:] -> that row's J values against col 0..N-1 in Hz
            if len(parts) < 3:
                break

            try:
                row_i = int(parts[0])
            except ValueError:
                break

            val_tokens = parts[2:]
            vals = []
            for tok in val_tokens:
                try:
                    vals.append(float(tok))
                except ValueError:
                    vals.append(np.nan)

            for j, v in enumerate(vals):
                if j < natm and np.isfinite(v):
                    Jmat[row_i, j] = v

            rows_parsed += 1

        if rows_parsed == 0:
            return (Jmat, False)

        # Symmetrize just in case the printout is upper- or lower-triangular
        Jmat = 0.5 * (Jmat + Jmat.T)
        return (Jmat, True)

    (J_full, ok) = _parse_J_table(ssc_text, natm)

    # ---- 3. Fallback: old tensor trace/3 heuristic on raw ----
    if not ok:
        LOG.debug(
            "SSC parse of printed J table failed; "
            "falling back to tensor trace/3 heuristic."
        )

        def _iso_from_tensor(val):
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 2 and arr.shape == (3, 3):
                # standard isotropic coupling = trace/3
                return float(np.trace(arr) / 3.0)
            if arr.ndim == 1 and arr.shape[0] == 3:
                return float(np.mean(arr))
            if arr.ndim == 0:
                return float(arr)
            return float(np.mean(arr))

        J_full = np.zeros((natm, natm), dtype=float)

        # Case A: raw is an (natm,natm) array already
        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            if raw.shape != (natm, natm):
                raise RuntimeError(
                    f"SSC kernel ndarray shape {raw.shape} "
                    f"!= ({natm},{natm}); cannot map."
                )
            J_full[:, :] = np.asarray(raw, dtype=float)

        # Case B: raw is (n_pairs,3,3) anisotropic tensors
        elif (
                isinstance(raw, np.ndarray)
                and raw.ndim == 3
                and raw.shape[1:] == (3, 3)
        ):
            n_pairs_expected = natm * (natm - 1) // 2
            if raw.shape[0] != n_pairs_expected:
                raise RuntimeError(
                    "SSC kernel ndarray shape "
                    f"{raw.shape} doesn't match {n_pairs_expected} pairs "
                    f"for natm={natm}."
                )
            k = 0
            for i in range(natm):
                for j in range(i + 1, natm):
                    Jij_iso_Hz = _iso_from_tensor(raw[k])
                    J_full[i, j] = Jij_iso_Hz
                    J_full[j, i] = Jij_iso_Hz
                    k += 1

        # Case C: raw is a dict {(ia,ja): tensor}
        elif isinstance(raw, dict):
            for (ia, ja), tensor_val in raw.items():
                Jij_iso_Hz = _iso_from_tensor(tensor_val)
                J_full[ia, ja] = Jij_iso_Hz
                J_full[ja, ia] = Jij_iso_Hz

        else:
            raise RuntimeError(
                f"SSC kernel() return type {type(raw)} not recognized."
            )

    # ---- 4. Subselect only the requested isotopes and label them ----
    #   - If isotopes_keep == ['1H'] (or ['h']), keep only hydrogens.
    #   - Otherwise use typical NMR-active nuclei.
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

    keep_idx = []
    keep_labels = []

    for (i, el) in enumerate(symbols):
        if only_H:
            if el.upper() == "H":
                keep_idx.append(i)
                keep_labels.append(f"H{i + 1}")
        else:
            guess_iso = iso_map.get(el.capitalize())
            if guess_iso in isotopes_keep:
                keep_idx.append(i)
                keep_labels.append(f"{el}{i + 1}")

    if not keep_idx:
        return (np.zeros((0, 0), dtype=float), [])

    sel = np.array(keep_idx, dtype=int)
    J_sel = J_full[np.ix_(sel, sel)].astype(float)

    return (J_sel, keep_labels)


def compute_sigma_J_and_energy_once(
        symbols: Sequence[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        xc: str,
        basis: str,
        use_gpu: bool,
        isotopes_keep: Sequence[str],
        need_J: bool,
        eps: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """
    One-stop heavy call for a single geometry.

    It does ALL of this with a single primary SCF:
      - run SCF (GPU if allowed) in the gas phase
      - clone that state onto a CPU MF (mf_cpu)
      - from mf_cpu:
          * get NMR shielding tensors -> per-atom σ_iso
          * get SSC -> dense J matrix in Hz (if need_J)
      - get energy:
          * if eps is None: just report gas-phase energy (Hartree)
            from the converged SCF
          * else: wrap mf_cpu in ddCOSMO(eps), run that once on CPU,
            and report the solvent-corrected total energy (Hartree)

    Returns:
        (sigma_iso, J_Hz, labels, e_tot)

        sigma_iso : (natm,) float
            isotropic shielding per atom

        J_Hz : (M,M) float
            symmetric scalar spin–spin couplings (Hz) for kept nuclei
            (0x0 if need_J == False)

        labels : list[str] length M
            nucleus labels ("H4", "C2", ...) matching rows/cols of J_Hz
            ([] if need_J == False)

        e_tot : float
            total electronic energy in Hartree.
            Gas-phase if eps is None,
            ddCOSMO(eps) if eps is not None.

    Why this exists:
        In f_predict_shifts_compute.py we were doing
        (1) SCF -> σ_iso,J
        (2) SCF again -> energy/PCM
        which doubles cost per cluster. Now callers can do it once.
    """
    # 1. Build molecule
    mol = make_mol(symbols, coords_A, charge, spin, basis)

    # 2. Run SCF ONCE (GPU if allowed)
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()  # gas-phase SCF

    # 3. Create a CPU MF seeded from that density
    mf_cpu = _property_on_cpu(mol, mf, xc)

    # ---- Shielding σ_iso from mf_cpu -------------------------------------
    arr = np.asarray(nmr_tensors_from_mf(mf_cpu))

    if arr.ndim == 3 and arr.shape[-1] == 3:
        # full 3x3 shielding tensors -> trace/3
        sigma_iso = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        # already isotropic per atom
        sigma_iso = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        # principal components given -> mean
        sigma_iso = arr.mean(axis=1).astype(float)
    else:
        # fallback: mean of diagonal elements
        sigma_iso = (
            np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)
        )

    # ---- Spin–spin J (Hz) from the SAME mf_cpu ---------------------------
    if need_J:
        (J_Hz, labels) = _build_J_matrix_Hz(
            mf_cpu,
            isotopes_keep=isotopes_keep,
        )
    else:
        J_Hz = np.zeros((0, 0), dtype=float)
        labels = []

    # ---- Energy (Hartree), with optional ddCOSMO -------------------------
    if eps is None:
        # vacuum energy from the converged SCF;
        # mf_cpu.e_tot should now reflect that SCF
        e_tot = float(mf_cpu.e_tot)
    else:
        # wrap mf_cpu with ddCOSMO and solve once on CPU
        mf_pcm = attach_pcm(mf_cpu, eps=eps)
        e_tot_pcm = mf_pcm.kernel()
        e_tot = float(e_tot_pcm)

    return (sigma_iso, J_Hz, labels, e_tot)


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
