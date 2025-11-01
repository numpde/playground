# /home/ra/repos/playground/20251031-MD-NMR/f_predict_shifts.py

"""
f_predict_shifts.py

Pipeline step f_...

Task
----
For each solute "tag" clustered in e_cluster/, do:
  1. Load each cluster's representative conformer PDB (cluster medoid).
  2. (Optionally) optimize that geometry with DFT + PCM solvent.
  3. Compute per-atom NMR shielding tensors (GIAO) using PySCF.
  4. Convert isotropic shieldings σ to chemical shifts δ (ppm)
     relative to TMS for 1H / 13C:
         δ_i = σ_ref(elem) - σ_i
     where σ_ref(elem) is the average shielding of that element
     in TMS computed at the SAME level.

  5. Write per-cluster shift tables.
  6. Compute a population-weighted "fast exchange" average.

Inputs (from e_cluster.py)
--------------------------
e_cluster/<tag>_clusters.tsv
    #cluster_id  count  fraction  medoid_frame_idx  rep_pdb_path
    fraction = cluster population fraction.
    rep_pdb_path = PDB path for that cluster's representative solute frame.

e_cluster/<tag>_cluster_<cid>_rep.pdb
    Representative solute conformer for cluster cid (aligned, heavy+H).

Outputs
-------
f_predict_shifts/<tag>_cluster_<cid>_shifts.tsv
    atom_idx  atom_name  element  sigma_iso  shift_ppm
    Per-atom predicted chemical shifts δ (ppm) for that cluster.

f_predict_shifts/<tag>_fastavg_shifts.tsv
    atom_idx  atom_name  element  shift_ppm_weighted
    Population-weighted average δ across clusters
    (fast-exchange assumption).

f_predict_shifts/<tag>_params.txt
    Provenance: functional, basis, PCM settings, SCF tolerance,
    geometry optimization toggle, etc.

Approximations / caveats
------------------------
- Charge/spin: we assume neutral closed-shell (charge=0, spin=0) unless
  overridden in _get_charge_spin_for_tag(). You *must* change that for
  charged states like deprotonated aspirin, etc.

- PCM solvent:
  We DO use PCM during geometry optimization (so we relax to a solvent-like
  geometry, e.g. DMSO-stabilized anion).
  BUT PySCF's NMR shielding code in pyscf.prop.nmr currently raises
  NotImplementedError for PCM-wrapped SCF objects.

  Workaround:
    * Optimize geometry with PCM (solvent_eps != None).
    * Then freeze that geometry.
    * Rebuild a *gas-phase* SCF (no PCM).
    * Run nmr.RKS/UKS(mf).kernel() on that gas-phase SCF.

  That means solvent affects geometry but not the electronic response
  at shielding time. This is a common pragmatic fallback when PCM+NMR
  isn't implemented.

- TMS reference:
  We generate TMS with RDKit, optimize it with PCM (same solvent_eps),
  then compute its shielding the same way: final shielding is computed
  in gas phase at that PCM-relaxed geometry.
  We average σ over all H in TMS → σ_ref["H"], and all C in TMS → σ_ref["C"].
  For other elements we currently output NaN.

- Fast vs slow exchange:
  We only produce a fast-exchange weighted average δ. If clusters are
  slow-exchanging on the NMR timescale, interpret cluster_<cid>_shifts.tsv
  individually instead of the fastavg table.

Usage
-----
    python f_predict_shifts.py

Requirements
------------
    pyscf            (core SCF/DFT and solvent.PCM)
    pyscf-properties (provides pyscf.prop.nmr)
    rdkit
    MDAnalysis
    numpy

Implementation notes
--------------------
- We keep SCF tolerances tight (conv_tol ~1e-9) because NMR shieldings are
  second-order response properties.
- BASIS is "def2-TZVP" because it's built-in and covers C/H/O/N/Si/etc.
  (pcS-2 is nicer for NMR but isn't bundled in your PySCF build.)

"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import MDAnalysis as mda
import numpy as np
from pyscf import gto, dft, solvent
# pyscf-properties extension (must be installed via pip from pyscf/properties)
from pyscf.prop import nmr as pyscf_nmr

# Kill "Module ... is under testing" spam from pyscf.prop.* modules
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module=r"pyscf\.prop\..*",
)

# optional geometry optimization via geomeTRIC
try:
    from pyscf.geomopt.geomeTRIC import optimize as geom_optimize

    GEOMOPT_AVAILABLE = True
except Exception:
    GEOMOPT_AVAILABLE = False

# ---------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------

DFT_XC = "b3lyp"  # hybrid DFT baseline, OK for 1H/13C NMR
BASIS = "def2-TZVP"  # built-in, good coverage incl Si (TMS), reasonable cost

PCM_METHOD = "IEF-PCM"  # implicit solvent model for geometry
PCM_EPS_DEFAULT = 46.7  # ~DMSO dielectric

SCF_CONV_TOL = 1e-9  # tighter SCF for response props
SCF_MAX_CYC = 200

OPTIMIZE_GEOMETRY = True  # turn off if you want raw MD conformer geometry

OUT_DIR = Path("f_predict_shifts")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _mkdir_p(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_charge_spin_for_tag(tag: str) -> Tuple[int, int]:
    """
    Return (charge, spin) for this solute tag.
    spin is 2S (0 = closed-shell singlet).
    Override here if the solute is an anion/cation or open-shell.

    NOTE: you *must* edit this for charged aspirin etc.
    """
    # Example for deprotonated aspirin anion:
    if "deprot" in tag.lower():
        return (-1, 0)

    return (0, 0)


def _pdb_to_symbols_names_coords(pdb_path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load a rep conformer PDB (solute only) and return:
        symbols  : ["C","H","O",...]
        names    : ["C1","H12",...] (PDB atom names)
        coords_A : (A,3) float64, Å
    """
    u = mda.Universe(str(pdb_path))
    ag = u.atoms
    coords_A = ag.positions.astype(float).copy()

    symbols: list[str] = []
    names: list[str] = []

    for atom in ag:
        el = getattr(atom, "element", None)
        if el is None or el == "":
            guess = (atom.name or "X")[0]
        else:
            guess = el
        symbols.append(str(guess).capitalize())
        names.append(atom.name or guess)

    return (symbols, names, coords_A)


def _rdkit_build_tms() -> Tuple[List[str], np.ndarray]:
    """
    Generate TMS ([Si](C)(C)(C)C), add hydrogens, embed 3D, do a quick UFF relax.
    Return (symbols, coords_A).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("[Si](C)(C)(C)C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    conf = mol.GetConformer()
    symbols: list[str] = []
    coords_list: list[Tuple[float, float, float]] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(idx)
        symbols.append(atom.GetSymbol().capitalize())
        coords_list.append((pos.x, pos.y, pos.z))

    coords_A = np.array(coords_list, dtype=float)
    return (symbols, coords_A)


def _make_mol(
        symbols: List[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
) -> "gto.Mole":
    """
    Build a PySCF Mole from symbols + Å coords.
    spin is 2S, so spin=0 = closed-shell.
    """
    atom_lines = [
        f"{sym} {x:.10f} {y:.10f} {z:.10f}"
        for (sym, (x, y, z)) in zip(symbols, coords_A)
    ]

    mol = gto.Mole()
    mol.atom = "\n".join(atom_lines)
    mol.unit = "Angstrom"
    mol.basis = BASIS
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol


def _attach_pcm_and_build_scf(
        mol,
        xc: str,
        solvent_eps: float | None,
        use_pcm: bool,
):
    """
    Build a DFT mean-field object (RKS or UKS), optionally wrap it in a PCM
    solvent model, set SCF tolerances, and *try* to move it to the GPU.

    Returns
    -------
    mf : PySCF / GPU4PySCF mean-field object
         - If GPU4PySCF is installed and a supported NVIDIA GPU is visible,
           this will be a GPU-backed object (arrays live on the GPU, SCF and
           gradients run on CUDA kernels).
         - Otherwise it falls back to normal CPU PySCF behavior.

    Why do we do GPU here?
    ----------------------
    - SCF, gradients, Hessians, PCM response, and geometry optimization are
      all GPU-accelerated in GPU4PySCF, including hybrid DFT functionals,
      with reported ~30× speedups vs a 32-core CPU node. :contentReference[oaicite:2]{index=2}
    - PySCF/GPU4PySCF exposes mf.to_gpu() / mf.to_cpu() to convert objects
      (and all their ndarray attributes) between CPU NumPy and GPU CuPy
      representations. :contentReference[oaicite:3]{index=3}
    - By doing the conversion here, every later call (mf.kernel(), geom_opt,
      mf.nuc_grad_method(), etc.) can benefit without any more code changes.
    """

    # 1. Build base mean-field: RKS if closed-shell, UKS if open-shell
    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = xc
    mf.conv_tol = SCF_CONV_TOL  # your tight SCF tolerance
    mf.max_cycle = SCF_MAX_CYC  # your SCF iteration cap

    # 2. Optionally wrap with PCM (for solvated optimization / SCF-in-solvent)
    if use_pcm and solvent_eps is not None:
        mf = solvent.PCM(mf)
        # PCM setup: PySCF/GPU4PySCF supports PCM and even PCM gradients on GPU. :contentReference[oaicite:4]{index=4}
        mf.with_solvent.method = PCM_METHOD
        mf.with_solvent.eps = solvent_eps
        mf.conv_tol = SCF_CONV_TOL
        mf.max_cycle = SCF_MAX_CYC

    # 3. Try to move to GPU
    #    .to_gpu() is provided by GPU4PySCF. If it's available and succeeds,
    #    mf becomes a GPU-backed object. If not (no GPU / no plugin), we
    #    just keep mf on CPU and continue normally.
    try:
        mf_gpu = mf.to_gpu()
        print("[info] using GPU4PySCF for SCF/DFT (+PCM if enabled)")
        mf = mf_gpu
    except Exception:
        print("[info] GPU4PySCF not available or GPU not supported; staying on CPU")

    return mf


def _optimize_geometry_if_enabled(
        symbols: List[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        solvent_eps: float | None,
) -> Tuple[List[str], np.ndarray]:
    """
    Optionally geometry-optimize this conformer with DFT+PCM.

    Returns (symbols_opt, coords_opt_A) in Å.

    If OPTIMIZE_GEOMETRY=False or geomeTRIC not available,
    we just return the input geometry.
    """
    if not OPTIMIZE_GEOMETRY or not GEOMOPT_AVAILABLE:
        return (symbols, coords_A)

    # Build Mole
    mol0 = _make_mol(symbols, coords_A, charge=charge, spin=spin)

    # Build SCF with PCM for geometry optimization
    mf0 = _attach_pcm_and_build_scf(
        mol=mol0,
        xc=DFT_XC,
        solvent_eps=solvent_eps,
        use_pcm=True,  # PCM ON for geom opt
    )

    # geomeTRIC.optimize returns a new Mole with optimized coords
    mol_opt = geom_optimize(mf0)

    coords_opt_A = mol_opt.atom_coords(unit="Angstrom")
    symbols_opt = [
        mol_opt.atom_symbol(i).capitalize()
        for i in range(mol_opt.natm)
    ]

    return (symbols_opt, coords_opt_A)


def _iso_from_tensors(tensors: np.ndarray) -> np.ndarray:
    """
    Convert shielding tensors → isotropic shielding σ_iso.
    Usually tensors.shape = (natm, 3, 3).
    σ_iso[i] = trace(tensor_i)/3.
    """
    arr = np.array(tensors)
    if arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] == 3:
        iso = np.trace(arr, axis1=1, axis2=2) / 3.0
        return iso.astype(float)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == 3:
        iso = np.trace(arr, axis1=0, axis2=1) / 3.0
        return iso.astype(float)
    # Fallback: diagonal mean
    return np.mean(np.diagonal(arr, axis1=-2, axis2=-1), axis=-1).astype(float)


def _compute_sigma_iso(
        symbols: List[str],
        coords_A: np.ndarray,
        charge: int,
        spin: int,
        solvent_eps: float | None,
) -> np.ndarray:
    """
    Compute isotropic shieldings σ_iso for one conformer.

    Steps:
      1. Build mol with (symbols, coords_A, charge, spin).
      2. Build gas-phase SCF (use_pcm=False), run mf.kernel().
         (We CANNOT pass PCM-wrapped mf to pyscf.prop.nmr: NotImplementedError.)
      3. Call pyscf.prop.nmr.RKS/UKS(mf).kernel() to get shielding tensors.
      4. Convert each tensor to σ_iso via trace/3.

    Returns:
      sigma_iso : (natm,) float64
    """
    mol = _make_mol(symbols, coords_A, charge=charge, spin=spin)

    # gas-phase SCF for NMR
    mf = _attach_pcm_and_build_scf(
        mol=mol,
        xc=DFT_XC,
        solvent_eps=solvent_eps,
        use_pcm=False,  # <-- IMPORTANT: NO PCM HERE
    )
    mf.kernel()

    # choose RKS vs UKS
    if mol.spin == 0:
        nmr_calc = pyscf_nmr.RKS(mf)
    else:
        nmr_calc = pyscf_nmr.UKS(mf)

    tensors = nmr_calc.kernel()
    sigma_iso = _iso_from_tensors(tensors)
    return sigma_iso


@lru_cache(maxsize=8)
def _tms_reference_sigma(
        solvent_eps: float | None,
) -> Dict[str, float]:
    """
    Compute {'H': σ_ref_H, 'C': σ_ref_C, ...} from TMS.

    We:
      - build TMS via RDKit
      - geometry-optimize with PCM (if enabled)
      - compute shieldings σ_iso using gas-phase NMR step
      - average per element

    We treat TMS as neutral singlet.
    """
    (tms_syms, tms_coords_A) = _rdkit_build_tms()

    (tms_syms_opt, tms_coords_opt_A) = _optimize_geometry_if_enabled(
        symbols=tms_syms,
        coords_A=tms_coords_A,
        charge=0,
        spin=0,
        solvent_eps=solvent_eps,
    )

    sigma_iso = _compute_sigma_iso(
        symbols=tms_syms_opt,
        coords_A=tms_coords_opt_A,
        charge=0,
        spin=0,
        solvent_eps=solvent_eps,
    )

    per_elem: dict[str, list[float]] = {}
    for (sym, sig) in zip(tms_syms_opt, sigma_iso):
        elem = sym.capitalize()
        per_elem.setdefault(elem, []).append(float(sig))

    ref_sigma: dict[str, float] = {}
    for (elem, vals) in per_elem.items():
        ref_sigma[elem] = float(np.mean(vals))

    return ref_sigma


def _sigma_to_delta_ppm(
        symbols: List[str],
        sigma_iso: np.ndarray,
        ref_sigma: Dict[str, float],
) -> np.ndarray:
    """
    Convert σ_iso[i] → δ ppm for each atom:
        δ_i = σ_ref(elem) - σ_iso[i]
    Only H and C currently get δ; others → NaN.
    """
    out: list[float] = []
    for (sym, sig) in zip(symbols, sigma_iso):
        elem = sym.capitalize()
        if elem in ref_sigma:
            out.append(ref_sigma[elem] - float(sig))
        else:
            out.append(np.nan)
    return np.array(out, dtype=float)


def _parse_clusters_tsv(tag: str) -> List[Dict[str, object]]:
    """
    Parse e_cluster/<tag>_clusters.tsv.
    Return list of:
        {
          'cid': int,
          'fraction': float,
          'rep_pdb': Path
        }
    """
    path = Path("e_cluster") / f"{tag}_clusters.tsv"
    rows: list[dict[str, object]] = []

    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            (
                cid_s,
                _count_s,
                frac_s,
                _medoid_frame_idx_s,
                rep_relpath,
            ) = line.rstrip("\n").split("\t")

            cid = int(cid_s)
            frac = float(frac_s)
            rep_pdb = Path("e_cluster") / rep_relpath

            rows.append({
                "cid": cid,
                "fraction": frac,
                "rep_pdb": rep_pdb,
            })

    if not rows:
        raise RuntimeError(f"{tag}: no valid clusters in {path}")

    return rows


def _write_cluster_shifts(
        out_dir: Path,
        tag: str,
        cid: int,
        atom_names: List[str],
        atom_symbols: List[str],
        sigma_iso: np.ndarray,
        delta_ppm: np.ndarray,
) -> Path:
    """
    Write per-cluster shifts table:
        atom_idx  atom_name  element  sigma_iso  shift_ppm
    """
    out_path = out_dir / f"{tag}_cluster_{cid}_shifts.tsv"
    with out_path.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tsigma_iso\tshift_ppm\n")
        for (i, (nm, el, sig, dppm)) in enumerate(
                zip(atom_names, atom_symbols, sigma_iso, delta_ppm)
        ):
            dppm_str = f"{dppm:.6f}" if np.isfinite(dppm) else "nan"
            fh.write(
                f"{i}\t{nm}\t{el}\t{sig:.6f}\t{dppm_str}\n"
            )
    return out_path


def _write_fastavg_shifts(
        out_dir: Path,
        tag: str,
        atom_names: List[str],
        atom_symbols: List[str],
        weighted_delta_ppm: np.ndarray,
) -> Path:
    """
    Write population-weighted fast-exchange averaged δ.
    """
    out_path = out_dir / f"{tag}_fastavg_shifts.tsv"
    with out_path.open("w") as fh:
        fh.write("# atom_idx\tatom_name\telement\tshift_ppm_weighted\n")
        for (i, (nm, el, dppm_w)) in enumerate(
                zip(atom_names, atom_symbols, weighted_delta_ppm)
        ):
            dppm_str = f"{dppm_w:.6f}" if np.isfinite(dppm_w) else "nan"
            fh.write(
                f"{i}\t{nm}\t{el}\t{dppm_str}\n"
            )
    return out_path


def _write_params(
        out_dir: Path,
        tag: str,
) -> Path:
    """
    Record provenance for audit / comparison to experiment.
    """
    out_path = out_dir / f"{tag}_params.txt"
    with out_path.open("w") as fh:
        fh.write(f"DFT_XC={DFT_XC}\n")
        fh.write(f"BASIS={BASIS}\n")
        fh.write(f"PCM_METHOD={PCM_METHOD}\n")
        fh.write(f"PCM_EPS_DEFAULT={PCM_EPS_DEFAULT}\n")
        fh.write(f"SCF_CONV_TOL={SCF_CONV_TOL}\n")
        fh.write(f"SCF_MAX_CYC={SCF_MAX_CYC}\n")
        fh.write(f"OPTIMIZE_GEOMETRY={OPTIMIZE_GEOMETRY}\n")
        fh.write(f"GEOMOPT_AVAILABLE={GEOMOPT_AVAILABLE}\n")
        fh.write("NOTE: shielding computed gas-phase at PCM-relaxed geometry\n")
    return out_path


def _process_tag(
        tag: str,
        solvent_eps: float | None = PCM_EPS_DEFAULT,
) -> None:
    """
    Driver for a single tag:
      - parse cluster metadata
      - compute σ_ref for TMS (cached)
      - for each cluster:
          * load rep conformer PDB
          * (optional) geom-opt with PCM
          * gas-phase shielding from that optimized geometry
          * δ ppm vs TMS
          * write <tag>_cluster_<cid>_shifts.tsv
      - write fast-exchange weighted δ
      - write params
    """
    print(f"[f_predict_shifts] tag={tag}")

    (charge, spin) = _get_charge_spin_for_tag(tag)
    clusters = _parse_clusters_tsv(tag)

    # Prepare TMS reference
    ref_sigma = _tms_reference_sigma(solvent_eps=solvent_eps)

    per_cluster_delta: list[np.ndarray] = []
    per_cluster_frac: list[float] = []

    atom_names_ref: list[str] | None = None
    atom_symbols_ref: list[str] | None = None

    for entry in clusters:
        cid = int(entry["cid"])
        frac = float(entry["fraction"])
        rep_pdb = entry["rep_pdb"]

        # load PDB
        (symbols_raw, atom_names, coords_raw_A) = _pdb_to_symbols_names_coords(rep_pdb)

        # geometry opt with PCM (if enabled)
        (symbols_opt, coords_opt_A) = _optimize_geometry_if_enabled(
            symbols=symbols_raw,
            coords_A=coords_raw_A,
            charge=charge,
            spin=spin,
            solvent_eps=solvent_eps,
        )

        # shielding at that geometry (gas-phase SCF)
        sigma_iso = _compute_sigma_iso(
            symbols=symbols_opt,
            coords_A=coords_opt_A,
            charge=charge,
            spin=spin,
            solvent_eps=solvent_eps,
        )

        # σ → δ ppm (H,C only)
        delta_ppm = _sigma_to_delta_ppm(
            symbols=symbols_opt,
            sigma_iso=sigma_iso,
            ref_sigma=ref_sigma,
        )

        _write_cluster_shifts(
            out_dir=OUT_DIR,
            tag=tag,
            cid=cid,
            atom_names=atom_names,
            atom_symbols=symbols_opt,
            sigma_iso=sigma_iso,
            delta_ppm=delta_ppm,
        )

        per_cluster_delta.append(delta_ppm)
        per_cluster_frac.append(frac)

        if atom_names_ref is None:
            atom_names_ref = atom_names
            atom_symbols_ref = symbols_opt

    # population-weighted fast-exchange:
    all_delta = np.stack(per_cluster_delta, axis=0)  # (k, A)
    fracs = np.array(per_cluster_frac, dtype=float)  # (k,)
    denom = np.sum(fracs)
    if denom <= 1e-15:
        weights = np.ones_like(fracs) / fracs.size
    else:
        weights = fracs / denom
    weighted_delta_ppm = np.sum(all_delta * weights[:, None], axis=0)

    _write_fastavg_shifts(
        out_dir=OUT_DIR,
        tag=tag,
        atom_names=atom_names_ref if atom_names_ref else [],
        atom_symbols=atom_symbols_ref if atom_symbols_ref else [],
        weighted_delta_ppm=weighted_delta_ppm,
    )

    _write_params(OUT_DIR, tag)

    print(f"[ok] {tag} clusters={len(clusters)}")


def main() -> None:
    """
    Discover all *_clusters.tsv under e_cluster/ and process each tag.
    Create output directory if needed.
    """
    _mkdir_p(OUT_DIR)

    cluster_files = sorted(Path("e_cluster").glob("*_clusters.tsv"))
    if not cluster_files:
        print("[error] no *_clusters.tsv in e_cluster/")
        return

    for cf in cluster_files:
        tag = cf.stem.removesuffix("_clusters")
        _process_tag(tag, solvent_eps=PCM_EPS_DEFAULT)

    print("[done]")


if __name__ == "__main__":
    main()
