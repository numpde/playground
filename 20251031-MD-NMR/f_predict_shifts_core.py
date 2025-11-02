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
    Build scalar J-coupling matrix (Hz) for the nuclei in `isotopes_keep`.

    Critical behavior:
    - We *force-verbose* the PySCF SpinSpinCoupling driver and temporarily
      replace its .stdout with an in-memory buffer so that PySCF will
      actually print the "Spin-spin coupling constant J (Hz)" table to us.
      This is necessary because our mf_cpu is usually quiet (verbose=0),
      so by default PySCF prints nothing.
    - We then parse that printed table as the authoritative Hz couplings.

    Fallbacks:
    - If we still can't parse a table, we fall back to decoding the raw
      return of .kernel() (pair tensors etc.), using nuc_pair if present.
      That branch is less reliable across PySCF versions and may not be in Hz.

    We return diagnostics so the caller can decide if values are usable.

    Returns
    -------
    (J_sel, keep_labels, diag)
        J_sel         np.ndarray (M,M), Hz
        keep_labels   list[str]  ["H4","H5", ...]
        diag          dict {
                           'raw_type': str(...),
                           'driver_attrs': [...],
                           'raw_ndarray_info' or 'raw_attrs': ...,
                           'parse_branch': str,
                           'J_full_info': {...},
                           'suspicious_zero': bool,
                       }
    """
    import io
    import contextlib
    import numpy as np

    natm = mf_cpu.mol.natm
    symbols = [atom[0] for atom in mf_cpu.mol._atom]  # ["C", "H", "H", ...]

    # ---- 1. Build SSC driver --------------------------------------------
    ssc_driver = _pick_ssc_driver(mf_cpu)

    # We'll need to run ssc_driver.kernel() while:
    #   - forcing verbosity so PySCF prints the Hz table
    #   - capturing that print (PySCF writes to obj.stdout, not always sys.stdout)
    # We'll restore original settings afterward.

    orig_verbose = getattr(ssc_driver, "verbose", None)
    orig_stdout = getattr(ssc_driver, "stdout", None)

    tmp_buf_driver = io.StringIO()   # we'll try to attach this to ssc_driver.stdout
    tmp_buf_redirect = io.StringIO() # fallback capture of sys.stdout if needed

    def _run_kernel_and_capture():
        """
        Run SpinSpinCoupling.kernel() with elevated verbosity and
        captured stdout. Return (raw, captured_text).
        """
        # Raise verbosity high enough to trigger the pretty J(Hz) print.
        try:
            if orig_verbose is not None:
                ssc_driver.verbose = max(int(orig_verbose), 4)
            else:
                ssc_driver.verbose = 4
        except Exception:
            # non-fatal: leave as-is
            pass

        captured_text = ""

        try:
            # Try to override the driver's .stdout to our own buffer.
            # Many PySCF property objects honor .stdout.
            ssc_driver.stdout = tmp_buf_driver
            # In this path, kernel() will (usually) write to tmp_buf_driver,
            # not to sys.stdout, so no need for redirect_stdout.
            raw_local = ssc_driver.kernel()
            captured_text = tmp_buf_driver.getvalue()

        except Exception:
            # If that didn't work, fall back to redirect_stdout() to at
            # least catch whatever lands on sys.stdout.
            with contextlib.redirect_stdout(tmp_buf_redirect):
                raw_local = ssc_driver.kernel()
            captured_text = tmp_buf_redirect.getvalue()

        return (raw_local, captured_text)

    try:
        (raw, ssc_text) = _run_kernel_and_capture()
    finally:
        # Restore original driver settings
        if orig_verbose is not None:
            try:
                ssc_driver.verbose = orig_verbose
            except Exception:
                pass
        if orig_stdout is not None:
            try:
                ssc_driver.stdout = orig_stdout
            except Exception:
                pass

    # ---- 2. Diagnostics setup -------------------------------------------
    diag: Dict[str, object] = {}
    diag["raw_type"] = str(type(raw))

    def _nd_summary(arr):
        arr_np = np.asarray(arr, dtype=float)
        return {
            "shape": tuple(arr_np.shape),
            "dtype": str(arr_np.dtype),
            "min": float(np.min(arr_np)) if arr_np.size else 0.0,
            "max": float(np.max(arr_np)) if arr_np.size else 0.0,
            "max_abs": float(np.max(np.abs(arr_np))) if arr_np.size else 0.0,
        }

    if isinstance(raw, np.ndarray):
        diag["raw_ndarray_info"] = _nd_summary(raw)
    else:
        diag["raw_attrs"] = [
            a for a in dir(raw) if not a.startswith("_")
        ]

    driver_attrs = [a for a in dir(ssc_driver) if not a.startswith("_")]
    diag["driver_attrs"] = driver_attrs

    # Emit preview of printed table for debugging (first ~30 lines)
    _preview_lines = "\n".join(ssc_text.splitlines()[:30])
    LOG.debug(
        "[_build_J_matrix_Hz] SSC stdout preview (truncated):\n%s\n[/preview]",
        _preview_lines,
    )

    # ---- 3. Strategy A: parse printed Hz table --------------------------
    def _parse_J_table(stdout_text, natm):
        """
        Parse a block like:

        Spin-spin coupling constant J (Hz)
                       #0        #1
                0 H    0.00000
                1 H  267.47280   0.00000

        into a full (natm,natm) float matrix (Hz).
        Return (Jmat, ok, branch_note).
        """
        Jmat = np.zeros((natm, natm), dtype=float)
        lines = stdout_text.splitlines()

        # Find header line containing "Spin-spin coupling constant" and "(Hz"
        start_idx = None
        for (li, line) in enumerate(lines):
            if "Spin-spin coupling constant" in line and "(Hz" in line:
                start_idx = li + 1
                break
        if start_idx is None:
            return (Jmat, False, "no_table_header")

        # After that header, PySCF may print one "    #0   #1 ..." line,
        # then the numeric rows.
        candidate_starts = (start_idx, start_idx + 1, start_idx + 2)
        parsed_any_numeric = False

        for cand_start in candidate_starts:
            if cand_start >= len(lines):
                continue

            tmp_J = np.zeros((natm, natm), dtype=float)
            tmp_numeric = False
            tmp_rows = 0

            for line in lines[cand_start:]:
                ls = line.strip()
                if not ls:
                    # blank -> table ended
                    break

                parts = ls.split()
                if len(parts) < 2:
                    # not a data row
                    break

                # first token should be row index (int)
                # e.g. "0", "1", maybe aligned/spaced.
                try:
                    row_i = int(parts[0])
                except ValueError:
                    # if it's not parseable as int, table probably ended
                    break

                # parts[1] is the element label ("H", "F", ...)
                float_tokens = parts[2:] if len(parts) > 2 else []

                vals: List[float] = []
                for tok in float_tokens:
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        vals.append(np.nan)

                for j, v in enumerate(vals):
                    if j < natm and np.isfinite(v):
                        tmp_J[row_i, j] = v
                        tmp_numeric = True
                tmp_rows += 1

            if tmp_numeric and tmp_rows > 0:
                parsed_any_numeric = True
                Jmat = tmp_J
                break

        if not parsed_any_numeric:
            return (Jmat, False, "table_parse_failed")

        # enforce symmetry
        Jmat = 0.5 * (Jmat + Jmat.T)
        return (Jmat, True, "table_ok")

    (J_full, ok_table, branch_note) = _parse_J_table(ssc_text, natm)
    if ok_table:
        diag["parse_branch"] = branch_note
        diag["J_full_info"] = _nd_summary(J_full)
        LOG.debug(
            "[_build_J_matrix_Hz] Parsed J from captured stdout table (%s). "
            "max|J|=%.6f Hz",
            branch_note,
            diag["J_full_info"]["max_abs"],
        )

    # ---- 4. Strategy B: fallback to raw if needed -----------------------
    if not ok_table:
        LOG.debug(
            "[_build_J_matrix_Hz] No usable stdout table (%s). "
            "Falling back to raw kernel() output of type %s",
            branch_note,
            type(raw),
        )

        def _iso_from_tensor(val):
            # take trace/3 of 3x3 tensor or average over components
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 2 and arr.shape == (3, 3):
                return float(np.trace(arr) / 3.0)
            if arr.ndim == 1 and arr.shape[0] == 3:
                return float(np.mean(arr))
            if arr.ndim == 0:
                return float(arr)
            return float(np.mean(arr))

        J_full = np.zeros((natm, natm), dtype=float)
        branch = "raw_unknown"

        # Case 1: ndarray (natm,natm)
        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            if raw.shape == (natm, natm):
                J_full[:, :] = np.asarray(raw, dtype=float)
                branch = "raw_ndarray_square"
            else:
                raise RuntimeError(
                    "SSC kernel ndarray shape %r != (%d,%d)"
                    % (raw.shape, natm, natm)
                )

        # Case 2: ndarray (n_pairs,3,3)
        elif (
            isinstance(raw, np.ndarray)
            and raw.ndim == 3
            and raw.shape[1:] == (3, 3)
        ):
            branch = "raw_pair_tensors"
            pairs_attr = getattr(ssc_driver, "nuc_pair", None)

            if pairs_attr is not None and len(pairs_attr) == raw.shape[0]:
                LOG.debug(
                    "[_build_J_matrix_Hz] Using ssc_driver.nuc_pair "
                    "to map pair tensors."
                )
                for k, pair in enumerate(pairs_attr):
                    try:
                        ia = int(pair[0])
                        ja = int(pair[1])
                    except Exception:
                        continue
                    Jij_iso = _iso_from_tensor(raw[k])
                    J_full[ia, ja] = Jij_iso
                    J_full[ja, ia] = Jij_iso
                branch = "raw_pair_tensors_nuc_pair"
            else:
                # fallback assume strict upper triangle ordering
                n_pairs_expected = natm * (natm - 1) // 2
                if raw.shape[0] != n_pairs_expected:
                    raise RuntimeError(
                        "SSC kernel ndarray shape %r doesn't match %d pairs "
                        "for natm=%d."
                        % (raw.shape, n_pairs_expected, natm)
                    )
                idx = 0
                for i in range(natm):
                    for j in range(i + 1, natm):
                        Jij_iso = _iso_from_tensor(raw[idx])
                        J_full[i, j] = Jij_iso
                        J_full[j, i] = Jij_iso
                        idx += 1

        # Case 3: dict {(ia,ja): tensor}
        elif isinstance(raw, dict):
            branch = "raw_dict_tensors"
            for (ia, ja), tensor_val in raw.items():
                Jij_iso = _iso_from_tensor(tensor_val)
                J_full[ia, ja] = Jij_iso
                J_full[ja, ia] = Jij_iso

        # Case 4: object attrs
        else:
            candidate_attrs = ["iso", "J", "j_hz", "j_ha", "j_au", "j_matrix"]
            for cand in candidate_attrs:
                if hasattr(raw, cand):
                    arr = np.asarray(getattr(raw, cand))
                    if arr.shape == (natm, natm):
                        J_full[:, :] = arr.astype(float)
                        branch = f"raw_attr_{cand}"
                        break

            if branch == "raw_unknown":
                for cand in candidate_attrs:
                    if hasattr(ssc_driver, cand):
                        arr = np.asarray(getattr(ssc_driver, cand))
                        if arr.shape == (natm, natm):
                            J_full[:, :] = arr.astype(float)
                            branch = f"driver_attr_{cand}"
                            break

        diag["parse_branch"] = branch
        diag["J_full_info"] = _nd_summary(J_full)
        LOG.debug(
            "[_build_J_matrix_Hz] Fallback branch '%s'. "
            "J_full max|J|=%.6f (raw units, assumed Hz or close).",
            branch,
            diag["J_full_info"]["max_abs"],
        )

    # ---- 5. sanity -------------------------------------------------------
    suspicious_zero = (diag["J_full_info"]["max_abs"] < 1e-3)
    diag["suspicious_zero"] = bool(suspicious_zero)
    if suspicious_zero:
        LOG.warning(
            "[_build_J_matrix_Hz] WARNING: Couplings are ~0 Hz everywhere "
            "(max|J| < 1e-3). This is chemically suspicious for 1H-1H "
            "couplings. parse_branch=%s raw_type=%s driver_attrs=%s",
            diag["parse_branch"],
            diag["raw_type"],
            diag.get("driver_attrs", []),
        )

    # ---- 6. isotope subselect -------------------------------------------
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
            guess_iso = iso_map.get(el.capitalize())
            if guess_iso in isotopes_keep:
                keep_idx.append(i)
                keep_labels.append(f"{el}{i + 1}")

    if not keep_idx:
        LOG.debug(
            "[_build_J_matrix_Hz] No nuclei matched isotopes_keep=%r; "
            "returning empty.",
            isotopes_keep,
        )
        return (np.zeros((0, 0), dtype=float), [], diag)

    sel = np.array(keep_idx, dtype=int)
    J_sel = J_full[np.ix_(sel, sel)].astype(float)

    LOG.debug(
        "[_build_J_matrix_Hz] Final subselected matrix: shape=%s, "
        "labels=%s, max|J|=%.6f Hz (suspicious_zero=%s)",
        J_sel.shape,
        keep_labels,
        float(np.max(np.abs(J_sel))) if J_sel.size else 0.0,
        diag["suspicious_zero"],
    )

    return (J_sel, keep_labels, diag)


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

    Steps:
      1. Build Mole from (symbols, coords_A, charge, spin, basis).
      2. Run SCF ONCE (GPU if allowed) in the gas phase.
      3. Clone that converged density into a pure-CPU mf_cpu.
      4. From mf_cpu:
         - NMR shielding tensors -> per-atom σ_iso
         - SSC -> dense J matrix in Hz (if need_J)
      5. Energy:
         - if eps is None: report mf_cpu.e_tot (gas-phase Hartree)
         - else: wrap mf_cpu in ddCOSMO(eps) and report that energy.

    Returns:
        (sigma_iso, J_Hz, labels, e_tot)

        sigma_iso : (natm,) float
            isotropic shieldings per atom

        J_Hz : (M,M) float
            spin–spin couplings (Hz) for kept nuclei,
            or 0x0 if need_J == False

        labels : list[str] length M
            nucleus labels ("H4", "C2", ...) matching rows/cols of J_Hz
            ([] if need_J == False)

        e_tot : float
            total electronic energy in Hartree.
            Gas-phase if eps is None,
            ddCOSMO(eps) if eps is not None.

    Why this exists:
        older code recomputed SCF separately for shifts vs energy.
        We unify it so each cluster geometry only pays once.
    """
    import numpy as np

    # 1. Build molecule
    mol = make_mol(symbols, coords_A, charge, spin, basis)

    # 2. Run SCF ONCE (GPU if allowed)
    mf = build_scf(mol, xc, use_gpu=use_gpu)
    _ = mf.kernel()  # gas-phase SCF

    # 3. Create a CPU MF seeded from that density
    mf_cpu = _property_on_cpu(mol, mf, xc)

    # 4a. Shielding tensors -> σ_iso
    tens = nmr_tensors_from_mf(mf_cpu)
    arr = np.asarray(tens, dtype=float)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        # full 3x3 shielding tensors -> trace/3
        sigma_iso = (np.trace(arr, axis1=1, axis2=2) / 3.0).astype(float)
    elif arr.ndim == 1:
        # already isotropic per atom
        sigma_iso = arr.astype(float)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        # principal components -> mean
        sigma_iso = arr.mean(axis=1).astype(float)
    else:
        # fallback: mean of diagonal elements
        sigma_iso = (
            np.diagonal(arr, axis1=-2, axis2=-1).mean(axis=-1).astype(float)
        )

    # 4b. Spin–spin J (Hz) from the SAME mf_cpu
    if need_J:
        (J_Hz, labels, diag_J) = _build_J_matrix_Hz(
            mf_cpu,
            isotopes_keep=isotopes_keep,
        )
        # propagate a strong hint into logs if it's garbage
        if diag_J.get("suspicious_zero", False):
            LOG.warning(
                "compute_sigma_J_and_energy_once(): SSC suspicious_zero=True "
                "(branch=%s raw_type=%s)",
                diag_J.get("parse_branch"),
                diag_J.get("raw_type"),
            )
    else:
        J_Hz = np.zeros((0, 0), dtype=float)
        labels = []

    # 5. Energy (Hartree), with optional ddCOSMO
    if eps is None:
        # gas-phase energy from mf_cpu
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

    SUCCESS criteria:
      - no exception is raised, and
      - we get a finite (possibly 1x1 or 2x2) J matrix
        for the requested nuclei, and
      - that matrix is not obviously "all ~0 Hz" garbage.

    Any RuntimeError raised here means "J is not reliably available".
    Callers use that to abort early if --require-j was set.
    """
    import numpy as np

    # tiny 2-proton test system
    symbols = ["H", "H"]
    coords_A = np.array(
        [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        dtype=float,
    )
    charge = 0
    spin = 0

    # SCF on CPU only
    mol = make_mol(symbols, coords_A, charge=charge, spin=spin, basis=basis)
    mf = build_scf(mol, xc, use_gpu=False)
    _ = mf.kernel()

    mf_cpu = _property_on_cpu(mol, mf, xc)

    (J_test, lbl_test, diag_test) = _build_J_matrix_Hz(
        mf_cpu,
        isotopes_keep=isotopes_keep,
    )

    # shape sanity
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

    # chemistry sanity
    if diag_test.get("suspicious_zero", False):
        raise RuntimeError(
            "SSC sanity check: all couplings ~0 Hz "
            f"(branch={diag_test.get('parse_branch')}, "
            f"raw_type={diag_test.get('raw_type')})"
        )

    # If we get here: SSC path is judged usable for production.
    LOG.debug(
        "SSC preflight OK: %d spins [%s], J matrix %s, branch=%s",
        len(lbl_test),
        ", ".join(lbl_test),
        J_test.shape,
        diag_test.get("parse_branch"),
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
        atom_names: Sequence[str],
) -> Tuple[Path, Path, Path]:
    """
    Write scalar spin–spin coupling info for this cluster:

        cluster_<cid>_J.npy
            dense symmetric (M,M) in Hz

        cluster_<cid>_J_labels.txt
            nucleus labels in matrix order (e.g. H4, H5, ...)

        cluster_<cid>_j_couplings.tsv
            upper triangle, human-readable with PDB names:
            i, j, label_i, pdb_i, label_j, pdb_j, J_Hz

    We enrich the TSV with the original PDB atom names so you can map
    H4 -> e.g. "H8x" from the input cluster geometry.

    We assume that each label in `labels` ends with an integer that
    matches the 1-based atom index in the SCF/PDB ordering:
        "H4" -> atom index 3 -> atom_names[3].

    If parsing fails, we fall back to "?" for the pdb_i/pdb_j column.
    """
    base_dir = out_dir / tag
    base_dir.mkdir(parents=True, exist_ok=True)

    def _label_to_pdb_name(lbl: str) -> str:
        # Extract trailing digits from lbl (e.g. "H12" -> "12")
        digits_rev: List[str] = []
        for ch in reversed(lbl):
            if ch.isdigit():
                digits_rev.append(ch)
            else:
                break
        if not digits_rev:
            return "?"
        digits = "".join(reversed(digits_rev))
        try:
            atom_idx_1based = int(digits)
        except ValueError:
            return "?"
        atom_idx_0based = atom_idx_1based - 1
        if 0 <= atom_idx_0based < len(atom_names):
            return str(atom_names[atom_idx_0based])
        return "?"

    # Dense J matrix
    npy_path = base_dir / f"cluster_{cid}_J.npy"
    np.save(npy_path, J_Hz.astype(float))

    # Labels file
    lbl_path = base_dir / f"cluster_{cid}_J_labels.txt"
    with lbl_path.open("w") as fh_lbl:
        for lbl in labels:
            fh_lbl.write(f"{lbl}\n")

    # Human TSV (upper triangle only)
    tsv_path = base_dir / f"cluster_{cid}_j_couplings.tsv"
    with tsv_path.open("w") as fh_tsv:
        fh_tsv.write("# i\tj\tlabel_i\tpdb_i\tlabel_j\tpdb_j\tJ_Hz\n")
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                lbl_i = labels[i]
                lbl_j = labels[j]
                pdb_i = _label_to_pdb_name(lbl_i)
                pdb_j = _label_to_pdb_name(lbl_j)
                Jij = float(J_Hz[i, j])
                fh_tsv.write(
                    f"{i}\t{j}\t{lbl_i}\t{pdb_i}\t{lbl_j}\t{pdb_j}\t{fmt(Jij)}\n"
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
