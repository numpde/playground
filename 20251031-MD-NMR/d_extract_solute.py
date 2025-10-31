#!/usr/bin/env python3
# /home/ra/repos/playground/20251031-MD-NMR/d_extract_solute.py

"""
d_extract_solute.py

Purpose
-------
Take each solvated production trajectory from `c_solvated` (solute + solvent,
periodic box, NPT @ 300 K / 1 bar) and collapse it down to just the solute.

For every system:
    1. Load <tag>_eq.pdb (topology + box) and <tag>.dcd (production frames).
    2. Identify the solute heavy atoms (not water / not bulk solvent).
       - Prefer a distinct non-solvent residue (e.g. the drug/ligand).
       - Fallback: largest bonded fragment that's < _MAX_SOLUTE_ATOMS.
    3. For each frame:
       - wrap that solute as a whole molecule back into the primary unit cell,
         using its center of geometry (so it's not split across PBC images).
       - align the wrapped coords to frame 0 using only heavy atoms,
         then apply that rigid transform to all atoms.
    4. Record:
       - aligned coordinates (Å) for all solute atoms
         → float32 array, shape (F, A, 3)
       - heavy-atom RMSD to frame 0 (Å) for sanity / clustering later
       - a "reference" PDB (frame 0 after wrap+align) for QC / QM input
       - a TSV with summary stats.

Outputs (written under ./d_extract_solute/):
    <tag>_solute_coords.npy   (F,A,3) float32 Å
    <tag>_rmsd.npy            (F,) float32 Å, heavy-atom RMSD vs frame0
    <tag>_solute_ref.pdb      frame-0 solute after wrap+align
    <tag>_diag.tsv            quick stats

Usage
-----
    python d_extract_solute.py
        auto-discovers c_*/*_eq.pdb (or *_eq_wrapped.pdb) + *.dcd

    python d_extract_solute.py path/to/sys_eq.pdb path/to/sys.dcd
        process just that pair
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import MDAnalysis as mda
from bugs import mkdir


# Output dir mirrors the other stages (a_init.py, c_solvated.py)
OUT_DIR = mkdir(Path(__file__).with_suffix(''))  # e.g. ./d_extract_solute/


# ---------------------------------------------------------------------
# constants / heuristics
# ---------------------------------------------------------------------

# Residue names we consider "bulk solvent", to avoid picking them as the solute.
# These cover water-like and common NMR solvents.
_SOLVENT_NAMES = {
    'HOH', 'H2O', 'WAT', 'SOL', 'TIP3', 'TIP3P',
    'DMS', 'DMSO', 'CL3', 'CDCL3', 'CHCL3',
}

# Safety ceiling: if a "candidate solute" has more atoms than this,
# it's probably the entire box (failed selection).
_MAX_SOLUTE_ATOMS = 200


# ---------------------------------------------------------------------
# PDB writer for the aligned reference frame
# ---------------------------------------------------------------------

def _write_solute_ref_pdb(out_path: Path, solute_atoms, coords_A0: np.ndarray) -> None:
    """
    Write a minimal single-residue PDB for the solute using the atom metadata
    from `solute_atoms` and coordinates coords_A0 (Å) from frame 0.

    We intentionally keep it simple:
    - one residue called "MOL", resid 1
    - ATOM records, occupancy=1.00, tempFactor=0.00
    This is good enough to open in PyMOL or feed to QM.
    """
    with out_path.open("w") as fh:
        fh.write("TITLE     solute reference frame 0\n")

        resname = "MOL"
        resid = 1

        for (idx, atom) in enumerate(solute_atoms):
            (x, y, z) = coords_A0[idx]
            serial = idx + 1  # PDB is 1-based
            atom_name = (atom.name or f"A{serial}")[:4].rjust(4)
            element = (getattr(atom, "element", None) or atom.name or "X")[0:2].rjust(2)

            fh.write(
                f"ATOM  {serial:5d} {atom_name} {resname:>3s} {resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
            )

        fh.write("TER\nEND\n")


# ---------------------------------------------------------------------
# solute selection / masks
# ---------------------------------------------------------------------

def _heavy_mask_from_atoms(atomgroup) -> np.ndarray:
    """
    Return a boolean mask [A] where True = heavy atom (not hydrogen).
    If .element is missing, fall back to first letter of atom.name.
    """
    mask = []
    for atom in atomgroup:
        sym = (getattr(atom, "element", None) or atom.name or "")[0].upper()
        mask.append(sym != "H")
    mask = np.array(mask, dtype=bool)
    if not np.any(mask):
        raise RuntimeError("Heavy-atom mask is empty (everything is H?)")
    return mask


def _largest_nonwater_residue(universe: mda.Universe):
    """
    Pick the residue with the most atoms that is not obviously solvent
    (not in _SOLVENT_NAMES). Works for "aspirin in water", "strychnine in DMSO", etc.
    """
    candidates: list[tuple[int, object]] = []
    for res in universe.residues:
        resname = (res.resname or "").upper()
        if resname not in _SOLVENT_NAMES:
            candidates.append((len(res.atoms), res))

    if not candidates:
        raise RuntimeError("No non-solvent residue found (cannot identify solute).")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]  # MDAnalysis.Residue


def _pick_solute_atoms(u: mda.Universe):
    """
    Return (solute_atoms, heavy_mask) for just the solute.

    Strategy:
    - If there are multiple residues: pick the largest residue whose
      name is not in _SOLVENT_NAMES (typical case once the solute is its
      own residue, and waters/solvent are separate residues).
    - If there's only one giant residue (bad topology export):
      fall back to bonded fragments and choose the largest plausible
      organic fragment (< _MAX_SOLUTE_ATOMS and has carbon).
    """
    residues = list(u.residues)

    # Case 1: normal multi-residue topology
    if len(residues) > 1:
        solute_res = _largest_nonwater_residue(u)
        solute_atoms = solute_res.atoms
        if solute_atoms.n_atoms > _MAX_SOLUTE_ATOMS:
            raise RuntimeError(
                f"Candidate solute residue has {solute_atoms.n_atoms} atoms "
                f"(>{_MAX_SOLUTE_ATOMS}). Looks like the whole box."
            )
        heavy_mask = _heavy_mask_from_atoms(solute_atoms)
        return (solute_atoms, heavy_mask)

    # Case 2: everything got dumped into one residue
    only_res = residues[0]
    all_atoms = only_res.atoms
    n_all = all_atoms.n_atoms

    # If it's already small, just take it directly.
    if n_all <= _MAX_SOLUTE_ATOMS:
        heavy_mask = _heavy_mask_from_atoms(all_atoms)
        return (all_atoms, heavy_mask)

    # Otherwise, try to split into fragments (MDAnalysis uses CONECT/bonds).
    frags = list(all_atoms.fragments)
    if len(frags) == 1:
        raise RuntimeError(
            "Single huge residue and only one bonded fragment. "
            "Can't isolate solute from solvent. "
            "You really need the wrapped per-residue PDB from c_solvated."
        )

    candidates = []
    for frag in frags:
        nat = frag.n_atoms
        if nat > _MAX_SOLUTE_ATOMS:
            continue
        elements = [
            (getattr(a, "element", None) or a.name or "")[0].upper()
            for a in frag
        ]
        has_carbon = ("C" in elements)
        # Skip 3-atom waters etc.
        if not has_carbon and nat <= 4:
            continue
        candidates.append((nat, frag))

    if not candidates:
        raise RuntimeError(
            "Could not find a reasonable fragment to treat as solute. "
            "Try exporting eq.pdb with distinct residues."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    solute_atoms = candidates[0][1]  # biggest plausible organic-ish fragment

    heavy_mask = _heavy_mask_from_atoms(solute_atoms)
    return (solute_atoms, heavy_mask)


# ---------------------------------------------------------------------
# geometry ops: PBC wrap, alignment, RMSD
# ---------------------------------------------------------------------

def _wrap_residue_by_center(coords_A: np.ndarray, box_lengths_A: np.ndarray) -> np.ndarray:
    """
    Translate a residue's coords (Å) so its center-of-geometry lies
    inside the primary [0, L) unit cell along each axis.

    Assumes an orthorhombic box (box_lengths_A ~ [Lx, Ly, Lz] in Å).
    """
    center = coords_A.mean(axis=0)  # Å
    # Compute integer box index of the center, subtract that box to bring COM in [0,L)
    shift = (center - np.floor(center / box_lengths_A) * box_lengths_A) - center
    return coords_A + shift


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix R (3x3) that best maps P -> Q
    in a least-squares sense. P, Q are both mean-centered (N,3) in Å.
    """
    C = P.T @ Q
    (V, _S, Wt) = np.linalg.svd(C)
    R = V @ Wt
    # Ensure a proper rotation (determinant +1), flip if needed.
    if np.linalg.det(R) < 0.0:
        V[:, -1] *= -1.0
        R = V @ Wt
    return R


def _align_to_reference(
    coords_A: np.ndarray,
    ref_coords_A: np.ndarray,
    heavy_mask: np.ndarray,
) -> np.ndarray:
    """
    Rigidly align coords_A (Å) to ref_coords_A (Å).

    Fit is computed on heavy atoms only, but the resulting rotation/translation
    is applied to all atoms.

    coords_A, ref_coords_A : (A,3) Å
    heavy_mask             : (A,) bool (True => include that atom in fit)
    returns aligned copy   : (A,3) Å
    """
    P = coords_A[heavy_mask, :]
    Q = ref_coords_A[heavy_mask, :]

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    R = _kabsch(Pc, Qc)

    full_center = coords_A.mean(axis=0, keepdims=True)
    ref_center = ref_coords_A.mean(axis=0, keepdims=True)

    aligned = (coords_A - full_center) @ R + ref_center
    return aligned


def _rmsd_heavy(frame_A: np.ndarray, ref_A: np.ndarray, heavy_mask: np.ndarray) -> float:
    """
    Heavy-atom RMSD between two already-aligned frames (Å).
    frame_A, ref_A : (A,3)
    heavy_mask     : (A,)
    """
    diff = frame_A[heavy_mask, :] - ref_A[heavy_mask, :]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# ---------------------------------------------------------------------
# core extraction routine
# ---------------------------------------------------------------------

def _process_one(tag: str, pdb_path: Path, dcd_path: Path, out_root: Path = OUT_DIR) -> None:
    """
    Extract the solute coordinates over a production trajectory and emit:
      <tag>_solute_coords.npy  (F,A,3) float32 Å
      <tag>_rmsd.npy           (F,) float32 Å heavy-atom RMSD vs frame0
      <tag>_solute_ref.pdb     representative conformer (frame0 after wrap+align)
      <tag>_diag.tsv           summary stats
    """
    print(f"[extract] {tag}")

    # 1) load trajectory (topology from *_eq.pdb, coords/time from *.dcd)
    u = mda.Universe(str(pdb_path), str(dcd_path))

    # 2) isolate solute atoms + heavy-atom mask
    (solute_atoms, heavy_mask) = _pick_solute_atoms(u)
    n_atoms = solute_atoms.n_atoms
    n_heavy = int(np.sum(heavy_mask))

    coords_over_time: list[np.ndarray] = []
    rmsd_trace: list[float] = []

    ref_coords_A: np.ndarray | None = None
    last_box_lengths_A: np.ndarray | None = None

    # track bounding-box spans after wrap+align (sanity check for "staying compact")
    spans_tracker: list[np.ndarray] = []

    # 3) iterate frames
    for (_frame_i, ts) in enumerate(u.trajectory):
        # MDAnalysis ts.dimensions ~ [lx, ly, lz, alpha, beta, gamma] in Å / deg
        box_lengths_A = np.array(ts.dimensions[:3], dtype=float)
        last_box_lengths_A = box_lengths_A  # stash most recent box vector lengths

        # copy current solute coords (Å)
        coords_A = solute_atoms.positions.copy()

        # wrap COM into primary unit cell so molecule isn't split across the box edge
        coords_wrapped_A = _wrap_residue_by_center(coords_A, box_lengths_A)

        # first frame defines reference
        if ref_coords_A is None:
            ref_coords_A = coords_wrapped_A.copy()
            aligned_A = ref_coords_A.copy()
            rmsd_val = 0.0
        else:
            aligned_A = _align_to_reference(
                coords_wrapped_A,
                ref_coords_A,
                heavy_mask,
            )
            rmsd_val = _rmsd_heavy(aligned_A, ref_coords_A, heavy_mask)

        coords_over_time.append(aligned_A.astype(np.float32))
        rmsd_trace.append(rmsd_val)

        # bounding box span of aligned solute
        mins = aligned_A.min(axis=0)
        maxs = aligned_A.max(axis=0)
        spans_tracker.append((maxs - mins))

    # stack results
    coords_arr = np.stack(coords_over_time, axis=0)        # (F,A,3) float32
    rmsd_arr = np.array(rmsd_trace, dtype=np.float32)      # (F,)
    spans_arr = np.stack(spans_tracker, axis=0)            # (F,3) Å
    span_mean = spans_arr.mean(axis=0)                     # (3,)
    span_max = spans_arr.max(axis=0)                       # (3,)

    (F, A, _) = coords_arr.shape

    # 4) save arrays
    npy_path = out_root / f"{tag}_solute_coords.npy"
    np.save(npy_path, coords_arr)

    rmsd_path = out_root / f"{tag}_rmsd.npy"
    np.save(rmsd_path, rmsd_arr)

    # 5) save reference solute PDB (frame 0 after wrap+align)
    pdb_out = out_root / f"{tag}_solute_ref.pdb"
    _write_solute_ref_pdb(
        out_path=pdb_out,
        solute_atoms=solute_atoms,
        coords_A0=coords_arr[0],
    )

    # 6) TSV diagnostics
    diag_path = out_root / f"{tag}_diag.tsv"
    with diag_path.open("w") as fh:
        fh.write("# key\tvalue\n")
        fh.write(f"tag\t{tag}\n")
        fh.write(f"frames\t{F}\n")
        fh.write(f"atoms_total\t{A}\n")
        fh.write(f"atoms_heavy\t{n_heavy}\n")
        if last_box_lengths_A is not None:
            fh.write(
                "box_lengths_A_last\t"
                f"{last_box_lengths_A[0]:.3f},"
                f"{last_box_lengths_A[1]:.3f},"
                f"{last_box_lengths_A[2]:.3f}\n"
            )
        fh.write(
            "span_mean_A\t"
            f"{span_mean[0]:.3f},"
            f"{span_mean[1]:.3f},"
            f"{span_mean[2]:.3f}\n"
        )
        fh.write(
            "span_max_A\t"
            f"{span_max[0]:.3f},"
            f"{span_max[1]:.3f},"
            f"{span_max[2]:.3f}\n"
        )
        fh.write(f"rmsd_mean_A\t{np.mean(rmsd_arr):.4f}\n")
        fh.write(f"rmsd_median_A\t{np.median(rmsd_arr):.4f}\n")
        fh.write(f"rmsd_max_A\t{np.max(rmsd_arr):.4f}\n")

    # 7) human-readable summary
    print(f"[ok] {tag}")
    print(f"     frames={F}, atoms={A} (heavy {n_heavy})")
    if last_box_lengths_A is not None:
        print(
            "     box_last(Å)=("
            f"{last_box_lengths_A[0]:.2f},"
            f"{last_box_lengths_A[1]:.2f},"
            f"{last_box_lengths_A[2]:.2f})"
        )
    print(
        "     span_mean(Å)=("
        f"{span_mean[0]:.2f},"
        f"{span_mean[1]:.2f},"
        f"{span_mean[2]:.2f}) "
        "span_max(Å)=("
        f"{span_max[0]:.2f},"
        f"{span_max[1]:.2f},"
        f"{span_max[2]:.2f})"
    )
    print(
        "     heavy-atom RMSD vs frame0: "
        f"mean={np.mean(rmsd_arr):.3f}Å "
        f"max={np.max(rmsd_arr):.3f}Å"
    )
    print(f"     wrote {npy_path.name}, {pdb_out.name}, {rmsd_path.name}, {diag_path.name}")


# ---------------------------------------------------------------------
# job discovery + CLI
# ---------------------------------------------------------------------

def _find_jobs() -> List[Tuple[str, Path, Path]]:
    """
    Auto-discover solvated systems produced by c_solvated.py.

    Looks under ./c_*/ for:
        *_eq.pdb            (or *_eq_wrapped.pdb for backward compat)
        <tag>.dcd           (production trajectory)

    Returns a list of (tag, pdb_path, dcd_path).
    """
    jobs: list[tuple[str, Path, Path]] = []

    pdb_candidates = list(Path(".").glob("c_*/*_eq_wrapped.pdb"))
    pdb_candidates += list(Path(".").glob("c_*/*_eq.pdb"))

    for pdb_path in sorted(pdb_candidates):
        stem = pdb_path.stem
        tag = (
            stem.removesuffix("_eq_wrapped")
                .removesuffix("_eq")
                .removesuffix("_wrapped")
        )

        # Prefer <tag>.dcd, but also allow <tag>*.dcd as a fallback
        dcd_list = list(pdb_path.parent.glob(f"{tag}.dcd"))
        if not dcd_list:
            dcd_list = list(pdb_path.parent.glob(f"{tag}*.dcd"))

        if not dcd_list:
            print(f"[skip] no DCD for {pdb_path}")
            continue

        # shortest filename wins (deterministic if multiple chunks exist)
        dcd_path = sorted(dcd_list, key=lambda p: len(p.name))[0]
        jobs.append((tag, pdb_path, dcd_path))

    return jobs


def main():
    # Explicit file pair mode
    if len(sys.argv) == 3:
        pdb_path = Path(sys.argv[1]).resolve()
        dcd_path = Path(sys.argv[2]).resolve()
        tag = (
            pdb_path.stem
            .removesuffix("_eq_wrapped")
            .removesuffix("_eq")
            .removesuffix("_wrapped")
        )
        _process_one(tag, pdb_path, dcd_path, OUT_DIR)
        return

    # Auto-discovery mode
    if len(sys.argv) == 1:
        jobs = _find_jobs()
        if not jobs:
            print("[error] no systems found under c_*/")
            return
        for (tag, pdb_path, dcd_path) in jobs:
            _process_one(tag, pdb_path, dcd_path, OUT_DIR)
        print("[done]")
        return

    # bad usage
    print("Usage:")
    print("  python d_extract_solute.py <eq.pdb> <traj.dcd>")
    print("  python d_extract_solute.py          # auto-discover in c_*/")
    sys.exit(1)


if __name__ == "__main__":
    main()
