# /home/ra/repos/playground/20251031-MD-NMR/f_predict_shifts.py
# Timestamp: 2025-11-01 16:00 Africa/Nairobi
#
# Summary
# -------
# Step f: predict NMR shifts (1H / 13C) for each solute "tag" clustered
# in e_cluster/ and write:
#   f_predict_shifts/<tag>_cluster_<cid>_shifts.tsv      (per-cluster)
#   f_predict_shifts/<tag>_fastavg_shifts.tsv            (population/Boltz)
#   f_predict_shifts/<tag>_params.txt                    (provenance)
#
# New in this revision
# --------------------
# - Temperature-aware Boltzmann reweighting:
#     weights_i(T) ∝ exp( -(G_i - G_min) / (k_B T) )
#   where G_i is approximated by a single-point electronic energy with PCM ON
#   at the (optionally PCM-relaxed) geometry. Toggle with:
#     --temp 298.15            (temperature in K; default 298.15)
#     --eps  46.7              (PCM dielectric; default 46.7 e.g., DMSO)
#     --no-boltz               (stick to MD cluster fractions)
#
# - Robust cluster table loader; graceful fallbacks for medoid PDB paths.
# - Clear, minimal logging; unchanged fast-exchange assumption.
#
# Mechanistic note
# ----------------
# This step assumes "fast exchange", so the observable shift is a population
# average over conformers:
#     <δ> = Σ_i w_i(T) · δ_i
# where δ_i are per-atom shifts from GIAO shielding and w_i are either:
#   (A) the MD cluster fractions (original behavior), or
#   (B) Boltzmann weights at temperature T (new; default ON).
#
# Caveats
# -------
# - PCM is used for geometry (if --opt) and for single-point energies G_i(T).
#   For NMR, PySCF's PCM wrapping is not supported by pyscf.prop.nmr, so
#   shieldings are computed from a gas-phase SCF at the frozen geometry.
# - Reference uses TMS computed at the same electronic level (gas-phase SCF).
# - If exchange slows at low T (k_ex ~ Δω), expect line splitting in real
#   spectra; inspect per-cluster TSVs rather than the fastavg.
#
# Source: user’s original f_predict_shifts.py (revised end-to-end).
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
# PySCF core
from pyscf import gto, dft
from pyscf.prop import nmr as pyscf_nmr

# Optional geometry optimization via geomeTRIC
try:
    from pyscf.geomopt.geomeTRIC import optimize as geom_optimize

    GEOMOPT_AVAILABLE = True
except Exception:
    GEOMOPT_AVAILABLE = False

# -----------------------------
# Tunables (can be CLI-overridden)
# -----------------------------
DFT_XC_DEFAULT = "b3lyp"
BASIS_DEFAULT = "def2-tzvp"
SCF_MAXCYC = 200
SCF_CONV_TOL = 1e-9
GRAD_CONV_TOL = 3e-4  # ~geomeTRIC default scale
NMR_NUCLEI = ("H", "C")  # only report 1H/13C

OUT_DIR = Path("f_predict_shifts")
CLUSTERS_DIR = Path("e_cluster")

PCM_EPS_DEFAULT = 46.7  # DMSO at ~298 K (you may choose to vary with T)
DO_GEOM_OPT_DEFAULT = True  # optimize medoids (PCM ON)
USE_BOLTZMANN_WEIGHTS_DEFAULT = True
TEMPERATURE_K_DEFAULT = 298.15


# -----------------------------
# Utilities
# -----------------------------
def _mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fmt(x: float) -> str:
    return f"{x:.6f}" if np.isfinite(x) else "nan"


# -----------------------------
# Chemistry helpers
# -----------------------------
def _make_mol(symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int) -> gto.Mole:
    mol = gto.Mole()
    mol.build(
        atom=[(s, tuple(r)) for s, r in zip(symbols, coords_A)],
        unit="Angstrom",
        basis=BASIS_DEFAULT,
        charge=charge,
        spin=spin,  # 2S; closed-shell: 0
        verbose=0,
    )
    return mol


def _build_rks_or_uks(mol: gto.Mole, xc: str) -> dft.rks.RKS | dft.uks.UKS:
    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = xc
    mf.max_cycle = SCF_MAXCYC
    mf.conv_tol = SCF_CONV_TOL
    return mf


def _attach_pcm_and_build_scf(mol: gto.Mole, xc: str, solvent_eps: Optional[float],
                              use_pcm: bool) -> dft.rks.RKS | dft.uks.UKS:
    mf = _build_rks_or_uks(mol, xc)
    if use_pcm and (solvent_eps is not None):
        # Simple dCOSMO-like wrapper; PySCF's `solvent` module auto-wraps MF.
        # NOTE: this MF cannot be passed into pyscf.prop.nmr
        try:
            from pyscf import solvent as pyscf_solvent  # lazy import
            mf = pyscf_solvent.ddCOSMO(mf)
            mf.with_solvent.eps = float(solvent_eps)
        except Exception as e:
            print(f"[warn] PCM requested but failed to attach: {e}; proceeding gas-phase.")
    return mf


def _optimize_geometry_pcm(symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int, xc: str,
                           solvent_eps: Optional[float]) -> np.ndarray:
    if not GEOMOPT_AVAILABLE:
        print("[info] geomeTRIC not available; skipping geometry optimization.")
        return coords_A

    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)
    mf_pcm = _attach_pcm_and_build_scf(mol, xc=xc, solvent_eps=solvent_eps, use_pcm=True)
    try:
        e_opt, mol_opt = geom_optimize(mf_pcm, tol=GRAD_CONV_TOL, maxsteps=200, callback=None)
        # Pull Angstrom coordinates back
        coords_bohr = mol_opt.atom_coords(unit="Bohr")
        coords_A_new = coords_bohr * 0.529177210903
        return coords_A_new
    except Exception as e:
        print(f"[warn] geometry optimization failed: {e}; using input geometry.")
        return coords_A


def _sigma_iso_from_tensors(tensors: np.ndarray) -> np.ndarray:
    # tensors shape: (natm, 3, 3); isotropic shielding = trace/3
    iso = np.trace(tensors, axis1=1, axis2=2) / 3.0
    return iso.astype(float)


def _compute_sigma_iso(symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int, xc: str) -> np.ndarray:
    # NMR requires gas-phase mf (PCM-wrapped MF not supported by pyscf.prop.nmr)
    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)
    mf = _build_rks_or_uks(mol, xc=xc)
    _ = mf.kernel()
    try:
        nmr = pyscf_nmr.NMR(mf)
        tensors = nmr.kernel()  # array per-atom (3x3)
        if isinstance(tensors, np.ndarray) and tensors.ndim == 3:
            return _sigma_iso_from_tensors(tensors)
    except NotImplementedError:
        pass  # fallthrough to diagonal mean below
    # Fallback: if something odd is returned, take diagonal mean per atom
    # (not expected for current PySCF, but keeps code robust)
    try:
        diag = np.diagonal(tensors, axis1=-2, axis2=-1).mean(axis=-1)
        return diag.astype(float)
    except Exception:
        raise RuntimeError("Unexpected NMR tensor format from PySCF.")


def _sigma_to_delta_ppm(symbols: Sequence[str], sigma_iso: np.ndarray, ref_sigma: Dict[str, float]) -> np.ndarray:
    # δ = σ_ref(element) - σ_i for 1H/13C; NaN for other elements
    out = np.full_like(sigma_iso, fill_value=np.nan, dtype=float)
    for i, el in enumerate(symbols):
        key = "H" if el.upper() == "H" else ("C" if el.upper() == "C" else None)
        if key is not None and key in ref_sigma:
            out[i] = ref_sigma[key] - sigma_iso[i]
    return out


def _tms_geometry() -> Tuple[List[str], np.ndarray]:
    # Minimal TMS geometry (Si(CH3)4) in Angstrom, tetrahedral; rough, good enough
    # for consistent referencing as long as level is the same across calls.
    # Si at origin; four methyl carbons roughly tetrahedral; hydrogens extended.
    # This is intentionally simple; exact geometry is not critical for a reference.
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []

    def add(atom: str, x: float, y: float, z: float) -> None:
        symbols.append(atom)
        coords.append((x, y, z))

    add("Si", 0.0, 0.0, 0.0)

    R_SiC = 1.86
    R_CH = 1.09
    # Tetrahedral directions
    dirs = [
        (1, 1, 1),
        (-1, -1, 1),
        (-1, 1, -1),
        (1, -1, -1),
    ]
    for dx, dy, dz in dirs:
        norm = (dx * dx + dy * dy + dz * dz) ** 0.5
        ux, uy, uz = dx / norm, dy / norm, dz / norm
        cx, cy, cz = R_SiC * ux, R_SiC * uy, R_SiC * uz
        add("C", cx, cy, cz)
        # Place three H's around each C in a trigonal pattern perpendicular to the C–Si bond
        # Quick orthonormal frame:
        ax, ay, az = -ux, -uy, -uz
        # arbitrary perpendicular vectors
        if abs(ax) < 0.9:
            px, py, pz = 1.0, 0.0, 0.0
        else:
            px, py, pz = 0.0, 1.0, 0.0
        # Gram-Schmidt
        vx, vy, vz = px - (ax * px + ay * py + az * pz) * ax, py - (ax * px + ay * py + az * pz) * ay, pz - (
                    ax * px + ay * py + az * pz) * az
        vnorm = (vx * vx + vy * vy + vz * vz) ** 0.5
        vx, vy, vz = vx / vnorm, vy / vnorm, vz / vnorm
        wx, wy, wz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx  # a×v

        for ang in (0.0, 2.0943951023931953, 4.1887902047863905):  # 0, 120°, 240°
            hx = cx + R_CH * (0.6 * ax + 0.8 * (math.cos(ang) * vx + math.sin(ang) * wx))
            hy = cy + R_CH * (0.6 * ay + 0.8 * (math.cos(ang) * vy + math.sin(ang) * wy))
            hz = cz + R_CH * (0.6 * az + 0.8 * (math.cos(ang) * vz + math.sin(ang) * wz))
            add("H", hx, hy, hz)

    return symbols, np.array(coords, dtype=float)


@lru_cache(maxsize=1)
def _tms_ref_sigma(xc: str) -> Dict[str, float]:
    # Build TMS, optionally PCM-relax it (same as solute), but final NMR stays gas-phase.
    symbols, coords_A = _tms_geometry()
    mol = _make_mol(symbols, coords_A, charge=0, spin=0)
    mf = _build_rks_or_uks(mol, xc=xc)
    _ = mf.kernel()
    nmr = pyscf_nmr.NMR(mf)
    tensors = nmr.kernel()
    sigma = _sigma_iso_from_tensors(tensors)
    # Return element-averaged σ_ref for H and C (averaging all H, all C in TMS)
    ref: Dict[str, float] = {}
    for el in ("H", "C"):
        sel = [i for i, s in enumerate(symbols) if s.upper() == el]
        ref[el] = float(np.mean(sigma[sel])) if sel else float("nan")
    return ref


# -----------------------------
# Energetics for Boltzmann weights
# -----------------------------
def _single_point_energy_pcm(symbols: Sequence[str], coords_A: np.ndarray, charge: int, spin: int, xc: str,
                             solvent_eps: Optional[float]) -> float:
    """PCM single-point electronic energy at fixed geometry (Hartree)."""
    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)
    mf = _attach_pcm_and_build_scf(mol, xc=xc, solvent_eps=solvent_eps, use_pcm=True)
    e_tot = float(mf.kernel())
    return e_tot


def _boltzmann_weights_from_energies(E_hartree: Sequence[float], temp_K: float) -> np.ndarray:
    """Return normalized Boltzmann weights at T for energy list in Hartree."""
    kB_Ha_per_K = 3.166811563e-6
    E = np.array(E_hartree, dtype=float)
    Emin = float(np.min(E))
    beta = 1.0 / (kB_Ha_per_K * temp_K)
    x = -beta * (E - Emin)
    x -= np.max(x)  # stabilize
    w = np.exp(x)
    return (w / np.sum(w)).astype(float)


# -----------------------------
# I/O: clusters and medoids
# -----------------------------
@dataclass
class ClusterRow:
    cid: int
    fraction: float
    medoid_pdb: Path


def _guess_medoid_path(tag: str, cid: int) -> Path:
    # Common patterns; adjust as needed to match your pipeline
    cand = [
        CLUSTERS_DIR / f"{tag}_cluster_{cid}_medoid.pdb",
        CLUSTERS_DIR / f"{tag}_cluster{cid}_medoid.pdb",
        CLUSTERS_DIR / f"{tag}_c{cid}_medoid.pdb",
    ]
    for p in cand:
        if p.exists():
            return p
    # Fallback: last resort — any PDB matching tag & cid
    for p in CLUSTERS_DIR.glob(f"*{tag}*{cid}*medoid*.pdb"):
        return p
    raise FileNotFoundError(f"No medoid PDB found for tag={tag} cid={cid}")


def _load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    rows: List[ClusterRow] = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = None
        for ln in reader:
            if not ln or ln[0].startswith("#"):
                continue
            if header is None and any(x in ln for x in ("cid", "fraction", "medoid")):
                header = [x.strip().lower() for x in ln]
                continue
            if header is None:
                # assume compact TSV: cid  fraction
                cid = int(ln[0])
                frac = float(ln[1])
                med = _guess_medoid_path(tag, cid)
            else:
                vals = {k: v for k, v in zip(header, ln)}
                cid = int(vals.get("cid") or vals.get("cluster_id"))
                frac = float(vals.get("fraction") or vals.get("frac") or vals.get("weight") or "0")
                medtxt = vals.get("medoid") or vals.get("pdb") or ""
                med = Path(medtxt) if medtxt else _guess_medoid_path(tag, cid)
            rows.append(ClusterRow(cid=cid, fraction=frac, medoid_pdb=med))
    if not rows:
        raise RuntimeError(f"Cluster table is empty: {path}")
    return rows


# -----------------------------
# Minimal PDB reader (no heavy deps)
# -----------------------------
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


# -----------------------------
# Charge/spin per tag (override as needed)
# -----------------------------
def _get_charge_spin_for_tag(tag: str) -> Tuple[int, int]:
    # Adjust for specific protonation/charge states by tag substring
    t = tag.lower()
    if "deprot" in t or "anion" in t:
        return (-1, 0)
    if "cation" in t or "prot" in t:
        return (+1, 0)
    # defaults
    return (0, 0)


# -----------------------------
# Writers
# -----------------------------
def _write_cluster_shifts(out_dir: Path, tag: str, cid: int, atom_names: Sequence[str], atom_symbols: Sequence[str],
                          sigma_iso: np.ndarray, delta_ppm: np.ndarray) -> Path:
    out_path = out_dir / f"{tag}_cluster_{cid}_shifts.tsv"
    with out_path.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tsigma_iso\tshift_ppm\n")
        for i, (nm, el, sig, dppm) in enumerate(zip(atom_names, atom_symbols, sigma_iso, delta_ppm)):
            fh.write(f"{i}\t{nm}\t{el}\t{_fmt(sig)}\t{_fmt(dppm)}\n")
    return out_path


def _write_fastavg_shifts(out_dir: Path, tag: str, atom_names: Sequence[str], atom_symbols: Sequence[str],
                          delta_ppm: np.ndarray) -> Path:
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


# -----------------------------
# Core per-tag processing
# -----------------------------
def _process_tag(
        tag: str,
        *,
        xc: str,
        basis: str,
        solvent_eps: Optional[float],
        do_geom_opt: bool,
        use_boltz: bool,
        temp_K: float,
) -> None:
    global BASIS_DEFAULT
    BASIS_DEFAULT = basis  # allow per-run basis override

    print(f"[tag] {tag}")

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
        atom_names, symbols, coords_A = _read_pdb_atoms(row.medoid_pdb)
        if do_geom_opt:
            coords_A = _optimize_geometry_pcm(symbols, coords_A, charge=charge, spin=spin, xc=xc,
                                              solvent_eps=solvent_eps)

        sigma_iso = _compute_sigma_iso(symbols, coords_A, charge=charge, spin=spin, xc=xc)
        delta_ppm = _sigma_to_delta_ppm(symbols, sigma_iso, ref_sigma)

        # Write per-cluster table
        _ = _write_cluster_shifts(OUT_DIR, tag, row.cid, atom_names, symbols, sigma_iso, delta_ppm)

        per_cluster_delta.append(delta_ppm)
        per_cluster_frac.append(float(row.fraction))

        if use_boltz:
            e_pcm = _single_point_energy_pcm(symbols, coords_A, charge=charge, spin=spin, xc=xc,
                                             solvent_eps=solvent_eps)
            per_cluster_energy.append(e_pcm)

        if atom_names_ref is None:
            atom_names_ref = list(atom_names)
            atom_symbols_ref = list(symbols)

    # Choose weights
    if use_boltz:
        weights = _boltzmann_weights_from_energies(per_cluster_energy, temp_K=temp_K)
    else:
        fracs = np.array(per_cluster_frac, dtype=float)
        denom = float(np.sum(fracs))
        weights = (fracs / denom) if denom > 1e-15 else (np.ones_like(fracs) / fracs.size)

    # Fast-exchange weighted average
    all_delta = np.stack(per_cluster_delta, axis=0)  # (k, A)
    w = weights[:, None]
    fastavg = np.sum(all_delta * w, axis=0)

    # Write outputs
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
            "geom_optimized_pcm": do_geom_opt,
            "weights": ("Boltzmann" if use_boltz else "MD fractions"),
            "temperature_K": temp_K,
            "k_B[Hartree/K]": 3.166811563e-6,
            "notes": "Shieldings from gas-phase SCF at fixed geometry; TMS ref at same level.",
        },
    )
    print(f"[ok] {tag}: {len(rows)} clusters → fastavg written.")


# -----------------------------
# CLI
# -----------------------------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute NMR shifts (fast-exchange average) from clustered conformers.")
    p.add_argument("--temp", type=float, default=TEMPERATURE_K_DEFAULT,
                   help="Temperature in Kelvin for Boltzmann weights (default 298.15).")
    p.add_argument("--eps", type=float, default=PCM_EPS_DEFAULT, help="PCM dielectric constant (default 46.7 ~ DMSO).")
    p.add_argument("--no-pcm", action="store_true", help="Disable PCM entirely (no geometry PCM, no energy PCM).")
    p.add_argument("--no-opt", action="store_true", help="Skip geometry optimization (use medoid geometry).")
    p.add_argument("--no-boltz", action="store_true", help="Use MD cluster fractions instead of Boltzmann weights.")
    p.add_argument("--xc", type=str, default=DFT_XC_DEFAULT, help="DFT functional (default b3lyp).")
    p.add_argument("--basis", type=str, default=BASIS_DEFAULT, help="Gaussian basis (default def2-tzvp).")
    p.add_argument("--tags", type=str, nargs="*",
                   help="Explicit tags to process (defaults to all *_clusters.tsv in e_cluster/).")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    _mkdir_p(OUT_DIR)

    solvent_eps = None if args.no_pcm else float(args.eps)
    do_geom_opt = (not args.no_opt)
    use_boltz = (not args.no_boltz)

    if args.tags:
        tags = args.tags
    else:
        cluster_files = sorted(CLUSTERS_DIR.glob("*_clusters.tsv"))
        if not cluster_files:
            print("[error] no *_clusters.tsv in e_cluster/")
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
        )
    print("[done]")


if __name__ == "__main__":
    main()
