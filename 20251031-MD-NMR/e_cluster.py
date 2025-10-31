#!/usr/bin/env python3
# /home/ra/repos/playground/20251031-MD-NMR/e_cluster.py
#
# Cluster solute conformations (e.g. aspirin in water) extracted by d_extract_solute.py.
#
# For each file like:
#   d_extract_solute/<tag>_solute_coords.npy     # (n_frames, n_atoms, 3) in Å, aligned
#   d_extract_solute/<tag>_solute_ref.pdb        # topology + ref coords
#
# We:
#   1. load coords
#   2. compute heavy-atom RMSD distances
#   3. run k-medoids for k = 2..5 (or less if not enough frames)
#   4. choose k with best silhouette
#   5. assign ALL frames to nearest medoid
#   6. write:
#        e_cluster/<tag>_cluster_<cid>_rep.pdb
#        e_cluster/<tag>_clusters.tsv
#        e_cluster/<tag>_labels.npy  (per-frame cluster id)
#
# This gives us conformer basins + their population %, which is what we'll
# weight in downstream NMR QM.

from pathlib import Path

import MDAnalysis as mda
import numpy as np
from bugs import mkdir


# ---------- helpers ----------

def _heavy_mask_from_universe_solute(universe):
    """
    Given a Universe built from <tag>_solute_ref.pdb, find the solute residue
    (largest non-solvent residue) and return:
        (solute_atoms, heavy_mask)
    where:
        solute_atoms is an AtomGroup,
        heavy_mask is boolean (n_atoms,) True = non-H atom.
    """
    water_like = {
        'HOH', 'H2O', 'WAT', 'SOL', 'TIP3', 'TIP3P',
        'DMS', 'DMSO', 'CL3', 'CDCL3', 'CHCL3'
    }

    # pick largest residue that's not in solvent list
    candidates = []
    for res in universe.residues:
        resname = (res.resname or "").upper()
        if resname not in water_like:
            candidates.append((len(res.atoms), res))
    if not candidates:
        raise RuntimeError("No non-solvent residue found in reference PDB.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    solute_res = candidates[0][1]

    solute_atoms = solute_res.atoms

    # Build heavy-atom mask that doesn't depend strictly on .element existing
    heavy_mask = np.array([
        (
                ((getattr(atom, "element", None) or atom.name or "")[0]).upper()
                != "H"
        )
        for atom in solute_atoms
    ], dtype=bool)

    return (solute_atoms, heavy_mask)


def _pairwise_rmsd_matrix(coords_A, heavy_mask):
    """
    Compute pairwise heavy-atom RMSD between frames.
    coords_A : (F, A, 3) in Å
    heavy_mask : (A,) bool, True for heavy atoms

    Returns:
        D : (F, F) float32
        where D[i,j] = RMSD(i,j) in Å
    """
    X = coords_A[:, heavy_mask, :]  # (F, H, 3) heavy atoms only
    # expand and diff
    diff = X[:, None, :, :] - X[None, :, :, :]  # (F,F,H,3)
    sq = (diff ** 2).sum(axis=-1)  # (F,F,H)
    mean_sq = sq.mean(axis=-1)  # (F,F)
    D = np.sqrt(mean_sq, dtype=np.float32)  # (F,F)
    return D


def _init_medoids_farthest_first(D, k):
    """
    Greedy farthest-first initialization for medoids.
    D : (F,F) distance matrix
    k : number of clusters
    Returns:
        medoid_idxs: list[int] length k
    """
    (F, _) = D.shape
    medoids = [0]  # start with frame 0
    while len(medoids) < k:
        # for each candidate frame, compute distance to nearest chosen medoid
        dist_to_nearest = np.min(D[:, medoids], axis=1)  # (F,)
        # pick the frame that maximizes that distance
        next_idx = int(np.argmax(dist_to_nearest))
        if next_idx in medoids:
            # fallback: pick any not used yet
            for cand in range(F):
                if cand not in medoids:
                    next_idx = cand
                    break
        medoids.append(next_idx)
    return medoids


def _assign_points(D, medoid_idxs):
    """
    Assign each frame to nearest medoid.
    D : (F,F) distance matrix
    medoid_idxs : list[int]
    Returns:
        labels : (F,) int cluster id in [0..k-1]
        dist_to_medoid : (F,) float
    """
    if len(medoid_idxs) == 1:
        labels = np.zeros(D.shape[0], dtype=int)
        dist_to = D[:, medoid_idxs[0]]
        return (labels, dist_to)

    # stack distances to each medoid
    dist_stack = np.stack([D[:, m] for m in medoid_idxs], axis=1)  # (F,k)
    labels = np.argmin(dist_stack, axis=1).astype(int)
    dist_to = dist_stack[np.arange(D.shape[0]), labels]
    return (labels, dist_to)


def _update_medoids(D, labels, k):
    """
    For each cluster, choose the point that minimizes total distance
    to all other points in that cluster (true k-medoids update).
    D : (F,F)
    labels : (F,)
    k : number clusters
    Returns:
        new_medoids : list[int]
    """
    new_medoids = []
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            # dead cluster; fallback to frame 0
            new_medoids.append(0)
            continue
        # submatrix of distances
        subD = D[np.ix_(members, members)]  # (m,m)
        # total distance for each candidate medoid to others in cluster
        total_dist = np.sum(subD, axis=1)  # (m,)
        best_local = members[int(np.argmin(total_dist))]
        new_medoids.append(best_local)
    return new_medoids


def _pam_kmedoids(D, k, max_iters=100):
    """
    Partitioning Around Medoids (k-medoids).
    Returns:
        medoid_idxs : list[int]
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


def _silhouette_score(D, labels, k):
    """
    Compute mean silhouette.
    D : (F,F) distances
    labels : (F,)
    k : number clusters
    Returns:
        float (higher is better, max ~1)
    """
    (F, _) = D.shape
    # precompute sets
    clusters = [np.where(labels == cid)[0] for cid in range(k)]

    s_vals = np.zeros(F, dtype=np.float32)

    for i in range(F):
        ci = labels[i]
        in_cluster = clusters[ci]

        # a(i): mean intra-cluster distance (excluding self)
        if len(in_cluster) > 1:
            a_i = np.mean(D[i, in_cluster[in_cluster != i]])
        else:
            # singleton cluster: define a_i = 0
            a_i = 0.0

        # b(i): lowest mean distance to another cluster
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


def _choose_k_and_cluster(D, k_min=2, k_max=5):
    """
    Try k in [k_min..k_max], pick the one with best silhouette.
    If we don't have enough frames to support k, skip that k.
    Returns:
        best_k, best_medoids, best_labels
    """
    (F, _) = D.shape

    # corner case: if F < 2 we cannot cluster
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

    # Fallback if everything failed somehow
    if best_k is None:
        best_k = 1
        best_medoids = [0]
        best_labels = np.zeros(F, dtype=int)

    return (best_k, best_medoids, best_labels)


def _assign_all_frames_to_medoids(all_coords_A, medoid_coords_A, heavy_mask):
    """
    Given medoid coords for a subset (shape (k,A,3)),
    assign ALL frames (shape (F,A,3)) to nearest medoid by heavy-atom RMSD.
    Returns:
        labels_all : (F,)
    """
    X = all_coords_A[:, heavy_mask, :]  # (F,H,3)
    M = medoid_coords_A[:, heavy_mask, :]  # (k,H,3)

    # dist[i,j] = RMSD(frame i, medoid j)
    diff = X[:, None, :, :] - M[None, :, :, :]  # (F,k,H,3)
    sq = (diff ** 2).sum(axis=-1)  # (F,k,H)
    mean_sq = sq.mean(axis=-1)  # (F,k)
    dist = np.sqrt(mean_sq)  # (F,k)

    labels_all = np.argmin(dist, axis=1).astype(int)
    return labels_all


def _write_cluster_reps(
        out_dir,
        tag,
        ref_pdb_path,
        rep_coords_list,
):
    """
    For each representative conformer (medoid), write a PDB.
    rep_coords_list : list of (A,3) arrays in Å, one per cluster
    """
    out_paths = []

    # We'll load the reference PDB topology once
    ref_u = mda.Universe(str(ref_pdb_path))
    (solute_atoms, _) = _heavy_mask_from_universe_solute(ref_u)

    for (cid, coords_A) in enumerate(rep_coords_list):
        if solute_atoms.n_atoms != coords_A.shape[0]:
            raise RuntimeError("Atom count mismatch writing cluster rep PDB.")

        solute_atoms.positions = coords_A
        out_path = out_dir / f"{tag}_cluster_{cid}_rep.pdb"
        solute_atoms.write(str(out_path))
        out_paths.append(out_path)

    return out_paths


# ---------- main ----------

def main():
    """
    Scan d_*/ *_solute_coords.npy, cluster each tag, and emit into e_cluster/.
    """

    base_dir = Path(".")
    out_dir = mkdir(Path(__file__).with_suffix(''))  # e_cluster/

    # find all coord sets
    coord_files = sorted(base_dir.glob("d_*/*_solute_coords.npy"))

    if not coord_files:
        print("No *_solute_coords.npy found under d_*/")
        return

    for coords_path in coord_files:
        # tag derivation: strip trailing _solute_coords
        tag = coords_path.stem.removesuffix("_solute_coords")

        # expect matching ref pdb next to it
        ref_pdb_path = coords_path.with_name(f"{tag}_solute_ref.pdb")
        if not ref_pdb_path.exists():
            print(f"[skip] {tag}: missing ref PDB {ref_pdb_path}")
            continue

        print(f"[cluster] {tag}")

        # load coords (F,A,3) Å float32/float64
        coords_all = np.load(coords_path)  # shape (F,A,3)

        # heavy atom mask from ref pdb
        ref_u = mda.Universe(str(ref_pdb_path))
        (solute_atoms_ref, heavy_mask) = _heavy_mask_from_universe_solute(ref_u)

        (F, A, _) = coords_all.shape
        if solute_atoms_ref.n_atoms != A:
            raise RuntimeError(f"{tag}: atom count mismatch between npy and ref PDB")

        # to limit O(F^2), subsample if huge
        MAX_FRAMES_FOR_CLUSTER = 2000
        if F > MAX_FRAMES_FOR_CLUSTER:
            # uniform stride
            stride = int(np.ceil(F / MAX_FRAMES_FOR_CLUSTER))
            idx_subset = np.arange(0, F, stride, dtype=int)
        else:
            idx_subset = np.arange(F, dtype=int)

        coords_sub = coords_all[idx_subset]  # (Fsub,A,3)

        # distance matrix on subset
        D = _pairwise_rmsd_matrix(coords_sub, heavy_mask)

        # choose k and cluster subset
        (best_k, medoid_subset_idxs, labels_subset) = _choose_k_and_cluster(D, k_min=2, k_max=5)

        # map medoid indices (subset space) -> true frame indices
        medoid_frame_idxs = [int(idx_subset[m]) for m in medoid_subset_idxs]

        # extract medoid coordinates in full frame indexing
        rep_coords_list = [coords_all[i] for i in medoid_frame_idxs]  # list[(A,3)]

        # assign ALL frames to nearest medoid
        medoid_coords_stack = np.stack(rep_coords_list, axis=0)  # (k,A,3)
        labels_all = _assign_all_frames_to_medoids(
            coords_all,
            medoid_coords_stack,
            heavy_mask,
        )  # (F,)

        # population stats
        pops = []
        for cid in range(best_k):
            count = int(np.sum(labels_all == cid))
            frac = count / F
            pops.append((cid, count, frac, medoid_frame_idxs[cid]))

        # write representative PDBs for each cluster
        rep_out_paths = _write_cluster_reps(
            out_dir=out_dir,
            tag=tag,
            ref_pdb_path=ref_pdb_path,
            rep_coords_list=rep_coords_list,
        )

        # write labels_all as .npy
        labels_path = out_dir / f"{tag}_labels.npy"
        np.save(labels_path, labels_all.astype(np.int16))

        # write summary TSV
        summary_path = out_dir / f"{tag}_clusters.tsv"
        with summary_path.open("w") as fh:
            fh.write("#cluster_id\tcount\tfraction\tmedoid_frame_idx\trep_pdb_path\n")
            for (cid, count, frac, midx), pdb_p in zip(pops, rep_out_paths):
                fh.write(
                    f"{cid}\t{count}\t{frac:.6f}\t{midx}\t{pdb_p.name}\n"
                )

        print(f"[ok] {tag}: k={best_k}, frames={F}, wrote")
        print(f"     {summary_path}")
        for p in rep_out_paths:
            print(f"     {p}")
        print(f"     {labels_path}")

    print("[done] all clustering complete.")


if __name__ == "__main__":
    main()
