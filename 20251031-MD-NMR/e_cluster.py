# /home/ra/repos/playground/20251031-MD-NMR/e_cluster.py

"""
e_cluster.py

Goal
----
Cluster solute conformations (aspirin, strychnine, etc.) after explicit-solvent MD.

Input per system `tag` (from d_extract_solute.py):
    d_extract_solute/<tag>_solute_coords.npy  # (F, A, 3) Å, solute only, wrapped+aligned
    d_extract_solute/<tag>_solute_ref.pdb     # topology + atom ordering for the solute
    d_extract_solute/<tag>_rmsd.npy           # (F,) Å heavy-atom RMSD vs frame0

What we do:
 1. Load coords and reference PDB.
 2. Build a heavy-atom mask.
 3. Estimate a decorrelation stride using the RMSD trace (so we don't
    cluster near-duplicate frames). Fallback: stride from max frame cap.
 4. Subsample frames with that stride.
 5. Compute pairwise heavy-atom RMSD matrix on the subsample.
 6. Run k-medoids (PAM) for k = 2..5, score by silhouette, pick best k.
 7. Use the resulting medoids as representative conformers.
 8. Assign *all* frames (full trajectory) to the nearest medoid
    to get per-frame cluster labels and cluster populations.
 9. Write out:
      e_cluster/<tag>_cluster_<cid>_rep.pdb   representative PDBs
      e_cluster/<tag>_clusters.tsv            populations, medoid frame idx
      e_cluster/<tag>_labels.npy              per-frame cluster labels
      e_cluster/<tag>_kinetics.tsv            simple label-to-label jump counts

Why kinetics?
-------------
If two clusters interconvert a lot (frequent A↔B jumps in time),
they're in fast exchange → NMR sees one averaged environment.
If transitions are extremely rare, they're slow exchange → NMR
may see separate peaks. We don't "solve" that here, but we record
the jump matrix so you can judge later.

Assumptions
-----------
- Frames in the .dcd (and therefore in *_solute_coords.npy) are saved
  at a fixed time interval. In c_solvated.py we save every REPORT_INT_STEPS
  (1000 MD steps at 1 fs/step) = ~1 ps/frame. Keep FRAME_DT_PS consistent.
- The first residue in *_solute_ref.pdb is the solute ("LIG"), and
  solvent was stripped already by d_extract_solute, so atom orders match.

Outputs go to ./e_cluster/.
"""

from pathlib import Path
from typing import List, Tuple

import MDAnalysis as mda
import numpy as np
from bugs import mkdir

# where to write results
OUT_DIR = mkdir(Path(__file__).with_suffix(''))  # ./e_cluster/

# we treat the following residue names as solvent (shouldn't happen here,
# but we keep them for safety in case the ref pdb wasn't stripped clean)
_SOLVENT_NAMES = {
    'HOH', 'H2O', 'WAT', 'SOL', 'TIP3', 'TIP3P',
    'DMS', 'DMSO', 'CL3', 'CDCL3', 'CHCL3'
}

# upper cap on how many frames to use when building the O(F^2) distance matrix
MAX_FRAMES_FOR_CLUSTER = 2000

# physical spacing between saved frames (ps/frame) in production DCD.
# this should match c_solvated.py's reporter stride (~1 ps/frame).
FRAME_DT_PS = 1.0


def _heavy_mask_from_universe_solute(universe: mda.Universe) -> Tuple[mda.core.groups.AtomGroup, np.ndarray]:
    """
    Given <tag>_solute_ref.pdb as a Universe, pick the "solute" residue
    (largest residue that's not water/solvent), and return:
      solute_atoms : AtomGroup
      heavy_mask   : (A,) bool, True if atom is non-H
    """
    candidates = []
    for res in universe.residues:
        resname = (res.resname or "").upper()
        if resname not in _SOLVENT_NAMES:
            candidates.append((len(res.atoms), res))

    if not candidates:
        raise RuntimeError("No non-solvent residue found in reference PDB.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    solute_res = candidates[0][1]
    solute_atoms = solute_res.atoms

    heavy_mask = np.array([
        (
                ((getattr(atom, "element", None) or atom.name or "")[0]).upper()
                != "H"
        )
        for atom in solute_atoms
    ], dtype=bool)

    if not np.any(heavy_mask):
        raise RuntimeError("Heavy-atom mask is empty.")
    return (solute_atoms, heavy_mask)


def _pairwise_rmsd_matrix(coords_A: np.ndarray, heavy_mask: np.ndarray) -> np.ndarray:
    """
    coords_A : (F, A, 3) Å
    heavy_mask : (A,) bool
    returns D : (F, F) float32, D[i,j] = heavy-atom RMSD(i,j) in Å
    """
    X = coords_A[:, heavy_mask, :]  # (F, H, 3)
    diff = X[:, None, :, :] - X[None, :, :, :]  # (F,F,H,3)
    sq = (diff ** 2).sum(axis=-1)  # (F,F,H)
    mean_sq = sq.mean(axis=-1)  # (F,F)
    D = np.sqrt(mean_sq, dtype=np.float32)  # (F,F)
    return D


def _init_medoids_farthest_first(D: np.ndarray, k: int) -> List[int]:
    """
    Greedy farthest-first init.
    """
    (F, _) = D.shape
    medoids = [0]
    while len(medoids) < k:
        dist_to_nearest = np.min(D[:, medoids], axis=1)
        next_idx = int(np.argmax(dist_to_nearest))
        if next_idx in medoids:
            for cand in range(F):
                if cand not in medoids:
                    next_idx = cand
                    break
        medoids.append(next_idx)
    return medoids


def _assign_points(D: np.ndarray, medoid_idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each frame to nearest medoid.
    returns:
      labels : (F,) int cluster id
      dist_to_medoid : (F,) float
    """
    if len(medoid_idxs) == 1:
        labels = np.zeros(D.shape[0], dtype=int)
        dist_to = D[:, medoid_idxs[0]]
        return (labels, dist_to)

    dist_stack = np.stack([D[:, m] for m in medoid_idxs], axis=1)  # (F,k)
    labels = np.argmin(dist_stack, axis=1).astype(int)
    dist_to = dist_stack[np.arange(D.shape[0]), labels]
    return (labels, dist_to)


def _update_medoids(D: np.ndarray, labels: np.ndarray, k: int) -> List[int]:
    """
    True k-medoids update step.
    """
    new_medoids = []
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            new_medoids.append(0)
            continue
        subD = D[np.ix_(members, members)]  # (m,m)
        total_dist = np.sum(subD, axis=1)  # (m,)
        best_local = members[int(np.argmin(total_dist))]
        new_medoids.append(best_local)
    return new_medoids


def _pam_kmedoids(D: np.ndarray, k: int, max_iters: int = 100) -> Tuple[List[int], np.ndarray]:
    """
    Partitioning Around Medoids.
    returns:
      medoid_idxs : list[int] length k
      labels      : (F,) int
    """
    medoid_idxs = _init_medoids_farthest_first(D, k)

    for _ in range(max_iters):
        (labels, _) = _assign_points(D, medoid_idxs)
        new_medoids = _update_medoids(D, labels, k)
        if set(new_medoids) == set(medoid_idxs):
            break
        medoid_idxs = new_medoids

    (labels, _) = _assign_points(D, medoid_idxs)
    return (medoid_idxs, labels)


def _silhouette_score(D: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Mean silhouette score.
    """
    (F, _) = D.shape
    clusters = [np.where(labels == cid)[0] for cid in range(k)]
    s_vals = np.zeros(F, dtype=np.float32)

    for i in range(F):
        ci = labels[i]
        in_cluster = clusters[ci]

        # a(i): mean intra-cluster distance
        if len(in_cluster) > 1:
            a_i = np.mean(D[i, in_cluster[in_cluster != i]])
        else:
            a_i = 0.0

        # b(i): nearest other-cluster mean distance
        b_i_candidates = []
        for cj, members in enumerate(clusters):
            if cj == ci or len(members) == 0:
                continue
            b_i_candidates.append(np.mean(D[i, members]))
        b_i = np.min(b_i_candidates) if b_i_candidates else 0.0

        denom = max(a_i, b_i)
        if denom <= 1e-12:
            s_vals[i] = 0.0
        else:
            s_vals[i] = (b_i - a_i) / denom

    return float(np.mean(s_vals))


def _choose_k_and_cluster(D: np.ndarray, k_min: int = 2, k_max: int = 5) -> Tuple[int, List[int], np.ndarray]:
    """
    Try multiple k, pick best silhouette.
    """
    (F, _) = D.shape
    if F < 2:
        return (1, [0], np.zeros(F, dtype=int))

    best_k = None
    best_score = -1.0
    best_medoids = None
    best_labels = None

    for k in range(k_min, k_max + 1):
        if k > F:
            break
        (medoids_k, labels_k) = _pam_kmedoids(D, k)
        score_k = _silhouette_score(D, labels_k, k)
        if score_k > best_score:
            best_score = score_k
            best_k = k
            best_medoids = medoids_k
            best_labels = labels_k

    if best_k is None:
        best_k = 1
        best_medoids = [0]
        best_labels = np.zeros(F, dtype=int)

    return (best_k, best_medoids, best_labels)


def _assign_all_frames_to_medoids(
        coords_all_A: np.ndarray,
        medoid_coords_A: np.ndarray,
        heavy_mask: np.ndarray,
) -> np.ndarray:
    """
    labels_all[i] = nearest medoid (heavy-atom RMSD).
    coords_all_A  : (F,A,3)
    medoid_coords_A : (k,A,3)
    """
    X = coords_all_A[:, heavy_mask, :]  # (F,H,3)
    M = medoid_coords_A[:, heavy_mask, :]  # (k,H,3)

    diff = X[:, None, :, :] - M[None, :, :, :]  # (F,k,H,3)
    sq = (diff ** 2).sum(axis=-1)  # (F,k,H)
    mean_sq = sq.mean(axis=-1)  # (F,k)
    dist = np.sqrt(mean_sq)  # (F,k)

    labels_all = np.argmin(dist, axis=1).astype(int)
    return labels_all


def _estimate_stride_from_rmsd(rmsd: np.ndarray) -> int:
    """
    Crude decorrelation stride:
    - compute normalized autocorr of the RMSD trace
    - pick first lag where autocorr < 0.2
    - floor at 1
    """
    F = rmsd.shape[0]
    if F < 4:
        return 1

    x = rmsd - np.mean(rmsd)
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return max(1, F // MAX_FRAMES_FOR_CLUSTER)

    # full autocorr up to, say, lag 2000 or F-2 (whichever smaller)
    max_lag = min(2000, F - 2)
    ac = []
    for lag in range(1, max_lag + 1):
        num = np.dot(x[:-lag], x[lag:])
        ac_val = num / denom
        ac.append(ac_val)
        if ac_val < 0.2:
            return lag

    # fallback if we never dipped below 0.2
    return max(1, F // MAX_FRAMES_FOR_CLUSTER)


def _pick_frame_indices(F: int, stride_guess: int) -> np.ndarray:
    """
    Turn stride guess into actual subset indices, also respecting
    MAX_FRAMES_FOR_CLUSTER.
    """
    stride = max(1, stride_guess)
    idx = np.arange(0, F, stride, dtype=int)
    if idx.size > MAX_FRAMES_FOR_CLUSTER:
        stride = int(np.ceil(F / MAX_FRAMES_FOR_CLUSTER))
        idx = np.arange(0, F, stride, dtype=int)
    return idx


def _transition_counts(labels_all: np.ndarray, k: int) -> np.ndarray:
    """
    Count label(i)->label(i+1) transitions.
    returns C (k,k) int
    """
    C = np.zeros((k, k), dtype=int)
    for (a, b) in zip(labels_all[:-1], labels_all[1:]):
        C[a, b] += 1
    return C


def _write_cluster_reps(
        out_dir: Path,
        tag: str,
        ref_pdb_path: Path,
        rep_coords_list: List[np.ndarray],
) -> List[Path]:
    """
    Write one PDB per cluster medoid.
    Each PDB is solute-only, same atom order as ref_pdb_path.
    """
    out_paths: List[Path] = []
    ref_u = mda.Universe(str(ref_pdb_path))
    (solute_atoms, _heavy_mask) = _heavy_mask_from_universe_solute(ref_u)

    for (cid, coords_A) in enumerate(rep_coords_list):
        if solute_atoms.n_atoms != coords_A.shape[0]:
            raise RuntimeError("Atom count mismatch in cluster rep write.")
        solute_atoms.positions = coords_A
        out_path = out_dir / f"{tag}_cluster_{cid}_rep.pdb"
        solute_atoms.write(str(out_path))
        out_paths.append(out_path)

    return out_paths


def _cluster_one(tag: str) -> None:
    """
    Load coords/rmsd for `tag`, cluster, and write outputs.
    """
    coords_path = Path(f"d_extract_solute/{tag}_solute_coords.npy")
    ref_pdb_path = Path(f"d_extract_solute/{tag}_solute_ref.pdb")
    rmsd_path = Path(f"d_extract_solute/{tag}_rmsd.npy")

    if not coords_path.exists() or not ref_pdb_path.exists():
        print(f"[skip] {tag}: missing coords or ref pdb")
        return
    if not rmsd_path.exists():
        print(f"[warn] {tag}: missing rmsd.npy (will fall back to naive stride)")

    coords_all = np.load(coords_path)  # (F,A,3) Å
    rmsd_trace = np.load(rmsd_path) if rmsd_path.exists() else None

    ref_u = mda.Universe(str(ref_pdb_path))
    (solute_atoms_ref, heavy_mask) = _heavy_mask_from_universe_solute(ref_u)

    (F, A, _) = coords_all.shape
    if solute_atoms_ref.n_atoms != A:
        raise RuntimeError(f"{tag}: atom count mismatch between coords and ref PDB")

    # decorrelate frames before building O(F^2) distance matrix
    if rmsd_trace is not None:
        stride_guess = _estimate_stride_from_rmsd(rmsd_trace)
    else:
        stride_guess = max(1, F // MAX_FRAMES_FOR_CLUSTER)

    idx_subset = _pick_frame_indices(F, stride_guess)
    coords_sub = coords_all[idx_subset]  # (Fsub,A,3)

    # distance matrix on subset
    D = _pairwise_rmsd_matrix(coords_sub, heavy_mask)

    # cluster subset
    (best_k, medoid_subset_idxs, labels_subset) = _choose_k_and_cluster(D, k_min=2, k_max=5)

    # medoid_subset_idxs are indices in coords_sub space -> map to full frame indices
    medoid_frame_idxs = [int(idx_subset[m]) for m in medoid_subset_idxs]

    # representative coordinates
    rep_coords_list = [coords_all[i] for i in medoid_frame_idxs]  # list[(A,3)]

    # assign ALL frames to nearest medoid
    medoid_coords_stack = np.stack(rep_coords_list, axis=0)  # (k,A,3)
    labels_all = _assign_all_frames_to_medoids(coords_all, medoid_coords_stack, heavy_mask)

    # population stats
    pops = []
    for cid in range(best_k):
        count = int(np.sum(labels_all == cid))
        frac = count / F
        pops.append((cid, count, frac, medoid_frame_idxs[cid]))

    # crude kinetics: transition counts matrix
    trans_counts = _transition_counts(labels_all, best_k)

    # write representatives
    rep_out_paths = _write_cluster_reps(
        out_dir=OUT_DIR,
        tag=tag,
        ref_pdb_path=ref_pdb_path,
        rep_coords_list=rep_coords_list,
    )

    # write per-frame labels
    labels_out = OUT_DIR / f"{tag}_labels.npy"
    np.save(labels_out, labels_all.astype(np.int16))

    # write population summary
    summary_path = OUT_DIR / f"{tag}_clusters.tsv"
    with summary_path.open("w") as fh:
        fh.write("#cluster_id\tcount\tfraction\tmedoid_frame_idx\trep_pdb_path\n")
        for (cid, count, frac, midx), pdb_p in zip(pops, rep_out_paths):
            fh.write(f"{cid}\t{count}\t{frac:.6f}\t{midx}\t{pdb_p.name}\n")

    # write simple kinetics summary
    kinetics_path = OUT_DIR / f"{tag}_kinetics.tsv"
    with kinetics_path.open("w") as fh:
        fh.write("# approx kinetics summary\n")
        fh.write(f"# FRAME_DT_PS={FRAME_DT_PS} ps per frame\n")
        fh.write("# rows: from_cluster, cols: to_cluster, value: jump count\n")
        for i in range(best_k):
            row = "\t".join(str(int(v)) for v in trans_counts[i])
            fh.write(f"{row}\n")

    print(f"[ok] {tag}: k={best_k}, frames={F}")
    print(f"     wrote {summary_path.name}, {labels_out.name}, {kinetics_path.name}")
    for p in rep_out_paths:
        print(f"     rep {p.name}")


def main():
    # auto-discover tags from d_extract_solute/*_solute_coords.npy
    coord_files = sorted(Path("d_extract_solute").glob("*_solute_coords.npy"))
    if not coord_files:
        print("[error] no *_solute_coords.npy in d_extract_solute/")
        return

    for coords_path in coord_files:
        tag = coords_path.stem.removesuffix("_solute_coords")
        print(f"[cluster] {tag}")
        _cluster_one(tag)

    print("[done]")


if __name__ == "__main__":
    main()
