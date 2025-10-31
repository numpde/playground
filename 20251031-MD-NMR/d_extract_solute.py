#!/usr/bin/env python3
# /home/ra/repos/playground/20251031-MD-NMR/d_extract_solute.py
#
# Extract just the solute (aspirin / strychnine) from solvated MD trajectories,
# wrap it into the primary periodic box, align all frames to frame 0,
# and write:
#   d_extract_solute/<tag>_solute_coords.npy   (n_frames, n_atoms, 3) Å
#   d_extract_solute/<tag>_solute_ref.pdb      (solute, frame 0 coords)
#
# Diagnostics (new):
#   d_extract_solute/<tag>_rmsd.npy            (n_frames,) heavy-atom RMSD vs frame 0 (Å)
#   d_extract_solute/<tag>_diag.tsv            summary stats
#
# Stdout per system:
#   frames, atoms, heavy atom count
#   bounding box span after wrapping/alignment
#   RMSD mean / max vs frame 0
#
# Usage:
#   python d_extract_solute.py
#   python d_extract_solute.py <eq_wrapped.pdb> <traj.dcd>

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import MDAnalysis as mda
from bugs import mkdir

# --- constants / heuristics -------------------------------------------------

_SOLVENT_NAMES = {
    'HOH', 'H2O', 'WAT', 'SOL', 'TIP3', 'TIP3P',
    'DMS', 'DMSO', 'CL3', 'CDCL3', 'CHCL3'
}


# --- helpers ----------------------------------------------------------------
_MAX_SOLUTE_ATOMS = 200  # sanity ceiling so we don't "select the whole box"

def _write_solute_ref_pdb(out_path: Path, solute_atoms, coords_A0: np.ndarray) -> None:
    """
    Write a minimal PDB containing ONLY the solute, using the atom
    metadata from `solute_atoms` and coordinates from coords_A0 (Å).

    We do not try to reconstruct bonds or residues perfectly.
    We just emit ATOM records with element/type info so you can load
    the representative conformer into PyMOL / QC software.

    out_path: Path to write
    solute_atoms: MDAnalysis AtomGroup (the solute we extracted)
    coords_A0: (A,3) numpy array in Å, frame 0 after wrap+align
    """
    with out_path.open("w") as fh:
        fh.write("TITLE     solute reference frame 0\n")

        # We'll write a single residue called MOL with resSeq 1
        resname = "MOL"
        resid = 1

        for (idx, atom) in enumerate(solute_atoms):
            (x, y, z) = coords_A0[idx]
            # Atom serials in PDB are 1-based
            serial = idx + 1
            # atom.name is usually like "C1", "H12", etc.
            atom_name = (atom.name or f"A{serial}")[:4].rjust(4)
            element = (getattr(atom, "element", None) or atom.name or "X")[0:2].rjust(2)

            # Classic PDB ATOM line, columns aligned.
            # We're not bothering with occupancy/tempFactor (set to 1.00, 0.00).
            fh.write(
                f"ATOM  {serial:5d} {atom_name} {resname:>3s} {resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
            )

        fh.write("TER\nEND\n")



def _pick_solute_atoms(u: mda.Universe):
    """
    Return (solute_atoms, heavy_mask) for the solute only, robust to two cases:

    Case 1: normal PDB with many residues (waters are HOH / TIP3, solute is MOL)
        → pick largest non-solvent residue.

    Case 2: ugly PDB with ONE giant residue containing everything
        → fall back to bonded fragments: pick the largest fragment
          that looks like a real molecule (< _MAX_SOLUTE_ATOMS atoms, not just water).

    Raises if we can't find something that looks like a single small molecule.
    """

    residues = list(u.residues)

    # --- Case 1: multiple residues → use residue heuristic
    if len(residues) > 1:
        solute_res = _largest_nonwater_residue(u)
        solute_atoms = solute_res.atoms
        if solute_atoms.n_atoms > _MAX_SOLUTE_ATOMS:
            raise RuntimeError(
                f"Candidate solute residue has {solute_atoms.n_atoms} atoms "
                f"(>{_MAX_SOLUTE_ATOMS}). Looks like whole box, abort."
            )
        heavy_mask = _heavy_mask_from_atoms(solute_atoms)
        return (solute_atoms, heavy_mask)

    # --- Case 2: single giant residue (your current situation)
    only_res = residues[0]
    all_atoms = only_res.atoms
    n_all = all_atoms.n_atoms

    # heuristic sanity: if it's already small, maybe it's actually just the solute PDB
    if n_all <= _MAX_SOLUTE_ATOMS:
        heavy_mask = _heavy_mask_from_atoms(all_atoms)
        return (all_atoms, heavy_mask)

    # otherwise: fragment fallback
    # MDAnalysis builds connectivity from CONECT records → .fragments splits by bonds.
    frags = list(all_atoms.fragments)
    if len(frags) == 1:
        raise RuntimeError(
            "Single huge residue and only one bonded fragment. "
            "Can't isolate solute from solvent. "
            "You really need the wrapped PDB output."
        )

    # score fragments:
    # - must be < _MAX_SOLUTE_ATOMS
    # - must have at least one carbon (to avoid picking a single water or bare ions)
    # pick the largest by atom count that passes filters
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
        if not has_carbon and nat <= 4:
            # 3-atom waters etc → skip
            continue
        candidates.append((nat, frag))

    if not candidates:
        raise RuntimeError(
            "Could not find a reasonable fragment to treat as solute. "
            "Need wrapped PDB where the solute is its own residue."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    solute_atoms = candidates[0][1]  # biggest plausible organic-ish fragment

    heavy_mask = _heavy_mask_from_atoms(solute_atoms)
    return (solute_atoms, heavy_mask)


def _largest_nonwater_residue(universe: mda.Universe):
    """
    Pick the residue with the most atoms that is not obviously solvent.
    Works for aspirin in water, strychnine in DMSO, etc.
    """
    candidates = []
    for res in universe.residues:
        resname = (res.resname or "").upper()
        if resname not in _SOLVENT_NAMES:
            candidates.append((len(res.atoms), res))

    if not candidates:
        raise RuntimeError("No non-solvent residue found (cannot identify solute).")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]  # MDAnalysis.Residue


def _wrap_residue_by_center(coords_A: np.ndarray, box_lengths_A: np.ndarray) -> np.ndarray:
    """
    Translate a residue's coords (Å) so its center-of-geometry lies in [0,L)
    along each axis. Assumes orthorhombic box, so box_lengths_A ~ [Lx,Ly,Lz] Å.

    coords_A:      (A,3) Å
    box_lengths_A: (3,) Å
    returns wrapped copy: (A,3) Å
    """
    center = coords_A.mean(axis=0)  # Å
    shift = (center - np.floor(center / box_lengths_A) * box_lengths_A) - center
    return coords_A + shift


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation R (3x3) that maps P -> Q in least squares sense.
    P, Q: (N,3) Å, both mean-centered.
    """
    C = P.T @ Q
    (V, S, Wt) = np.linalg.svd(C)
    R = V @ Wt
    if np.linalg.det(R) < 0.0:
        # fix improper rotation
        V[:, -1] *= -1.0
        R = V @ Wt
    return R


def _align_to_reference(
        coords_A: np.ndarray,
        ref_coords_A: np.ndarray,
        heavy_mask: np.ndarray
) -> np.ndarray:
    """
    Rigidly align coords_A (Å) to ref_coords_A (Å) using only heavy atoms
    to compute the rotation, then apply that rotation to all atoms.

    coords_A, ref_coords_A: (A,3) Å
    heavy_mask:             (A,) bool (True => include in fit)
    returns aligned copy:   (A,3) Å
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


def _heavy_mask_from_atoms(atomgroup) -> np.ndarray:
    """
    Boolean mask [A] where True means "not hydrogen".
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


def _rmsd_heavy(frame_A: np.ndarray, ref_A: np.ndarray, heavy_mask: np.ndarray) -> float:
    """
    Heavy-atom RMSD between two aligned frames (Å).
    frame_A, ref_A: (A,3)
    heavy_mask:     (A,)
    """
    diff = frame_A[heavy_mask, :] - ref_A[heavy_mask, :]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _process_one(tag: str, pdb_path: Path, dcd_path: Path, out_root: Path) -> None:
    """
    Extract solute coords over trajectory, align, and emit diagnostics.

    Writes:
      <tag>_solute_coords.npy  (frames x atoms x 3) Å
      <tag>_solute_ref.pdb
      <tag>_rmsd.npy           per-frame heavy-atom RMSD vs frame0 (Å)
      <tag>_diag.tsv           summary stats
    """

    print(f"[extract] {tag}")

    # --- load trajectory
    u = mda.Universe(str(pdb_path), str(dcd_path))

    # --- pick solute residue
    (solute_atoms, heavy_mask) = _pick_solute_atoms(u)
    n_atoms = solute_atoms.n_atoms
    n_heavy = int(np.sum(heavy_mask))

    coords_over_time: list[np.ndarray] = []
    rmsd_trace: list[float] = []

    ref_coords_A: np.ndarray | None = None
    last_box_lengths_A: np.ndarray | None = None

    # Track bounding box spans (min/max per frame after wrap+align)
    spans_tracker = []  # list of (span_x, span_y, span_z) in Å

    for (frame_i, ts) in enumerate(u.trajectory):
        # assume orthorhombic box
        box_lengths_A = np.array(ts.dimensions[:3], dtype=float)
        last_box_lengths_A = box_lengths_A  # remember last seen box length

        # copy current solute coords (Å)
        coords_A = solute_atoms.positions.copy()

        # wrap COM into primary unit cell so molecule isn't split
        coords_wrapped_A = _wrap_residue_by_center(coords_A, box_lengths_A)

        if ref_coords_A is None:
            # first frame defines reference geometry
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

        # bounding box of aligned solute (sanity: should be small, e.g. < ~15 Å)
        mins = aligned_A.min(axis=0)
        maxs = aligned_A.max(axis=0)
        spans_tracker.append((maxs - mins))

    coords_arr = np.stack(coords_over_time, axis=0)  # (F,A,3) float32
    rmsd_arr = np.array(rmsd_trace, dtype=np.float32)  # (F,)
    spans_arr = np.stack(spans_tracker, axis=0)  # (F,3) Å
    span_mean = spans_arr.mean(axis=0)  # (3,)
    span_max = spans_arr.max(axis=0)  # (3,)

    (F, A, _) = coords_arr.shape


    # --- save coords array and RMSD trace
    npy_path = out_root / f"{tag}_solute_coords.npy"
    np.save(npy_path, coords_arr)

    rmsd_path = out_root / f"{tag}_rmsd.npy"
    np.save(rmsd_path, rmsd_arr)

    # --- save reference solute PDB (frame 0 coords)
    pdb_out = out_root / f"{tag}_solute_ref.pdb"
    _write_solute_ref_pdb(
        out_path=pdb_out,
        solute_atoms=solute_atoms,
        coords_A0=coords_arr[0],
    )


    # --- write diagnostics TSV
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

    # --- human-readable summary to stdout
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
        f"     heavy-atom RMSD vs frame0: "
        f"mean={np.mean(rmsd_arr):.3f}Å "
        f"max={np.max(rmsd_arr):.3f}Å"
    )
    print(f"     wrote {npy_path.name}, {pdb_out.name}, {rmsd_path.name}, {diag_path.name}")


# --- main -------------------------------------------------------------------

def _find_jobs() -> List[Tuple[str, Path, Path]]:
    """
    (unchanged)
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

        dcd_list = list(pdb_path.parent.glob(f"{tag}.dcd"))
        if not dcd_list:
            dcd_list = list(pdb_path.parent.glob(f"{tag}*.dcd"))

        if not dcd_list:
            print(f"[skip] no DCD for {pdb_path}")
            continue

        dcd_path = sorted(dcd_list, key=lambda p: len(p.name))[0]
        jobs.append((tag, pdb_path, dcd_path))

    return jobs


def main():
    out_root = mkdir(Path(__file__).with_suffix(''))  # e.g. d_extract_solute/

    # Explicit mode
    if len(sys.argv) == 3:
        pdb_path = Path(sys.argv[1]).resolve()
        dcd_path = Path(sys.argv[2]).resolve()
        tag = (
            pdb_path.stem
            .removesuffix("_eq_wrapped")
            .removesuffix("_eq")
            .removesuffix("_wrapped")
        )
        _process_one(tag, pdb_path, dcd_path, out_root)
        return

    # Auto mode
    if len(sys.argv) == 1:
        jobs = _find_jobs()
        if not jobs:
            print("[error] no systems found under c_*/")
            return
        for (tag, pdb_path, dcd_path) in jobs:
            _process_one(tag, pdb_path, dcd_path, out_root)
        print("[done]")
        return

    # bad usage
    print("Usage:")
    print("  python d_extract_solute.py <eq_wrapped.pdb> <traj.dcd>")
    print("  python d_extract_solute.py   # auto-discover in c_*/")
    sys.exit(1)


if __name__ == "__main__":
    main()
