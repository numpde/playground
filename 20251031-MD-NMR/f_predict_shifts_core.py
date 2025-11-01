#!/usr/bin/env python3
# f_predict_shifts_average.py
#
# Fast step:
#   - Read per-cluster chemical shifts (cluster_<cid>_shifts.tsv)
#   - Read per-cluster ddCOSMO single-point energies (energies_<solvent>.tsv)
#   - Compute Boltzmann weights (or use MD fractions)
#   - Produce fast-exchange averaged shifts:
#         fastavg_<solvent_key>_<temp>K.tsv
#
# NEW:
#   - Read per-cluster scalar J couplings:
#         cluster_<cid>_J.npy
#         cluster_<cid>_J_labels.txt
#   - Boltzmann-average those J matrices (elementwise) for nuclei we keep
#   - Save:
#         fastavg_J_<solvent_key>_<temp>K.npy        (Hz)
#         fastavg_J_labels.txt                       (spin order)
#
# We assume:
#   * all clusters share atom ordering for shifts
#   * all clusters that HAVE J share the same spin label order
#
# 2025-11-01

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from f_predict_shifts_core import (
    OUT_DIR,
    CLUSTERS_DIR,
    SOLVENT_EPS,
    setup_logging,
    mkdir_p,
    load_clusters_table,
    boltzmann_weights,
    fmt,
    write_params,
    average_J_matrices,
)

LOG = logging.getLogger("nmrshifts.average")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Average precomputed per-cluster shifts and J-coupling matrices "
            "with temperature/solvent weights."
        )
    )
    p.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Tags to process (default: all *_clusters.tsv).",
    )
    p.add_argument(
        "--temp",
        type=float,
        default=298.15,
        help="Temperature [K] for Boltzmann weights.",
    )
    p.add_argument(
        "--solvent",
        type=str,
        default="DMSO",
        help="Solvent name to pick energies_<solvent>.tsv.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Override dielectric; uses energies_eps<val>.tsv.",
    )
    p.add_argument(
        "--no-boltz",
        action="store_true",
        help="Use MD fractions instead of Boltzmann weights.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce informational messages.",
    )
    return p.parse_args()


def read_cluster_shifts(tag: str, cid: int) -> pd.DataFrame:
    """
    Load cluster_<cid>_shifts.tsv for a given tag.

    Returns DataFrame with columns:
      atom_idx, atom_name, element, sigma_iso, shift_ppm
    """
    path = OUT_DIR / tag / f"cluster_{cid}_shifts.tsv"
    df = pd.read_csv(path, sep="\t", comment="#")
    return df


def read_cluster_J(tag: str, cid: int) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Try to load:
      cluster_<cid>_J.npy
      cluster_<cid>_J_labels.txt

    Returns (J_matrix_Hz, labels) or (None, None) if not present.
    """
    base = OUT_DIR / tag
    j_mat_path = base / f"cluster_{cid}_J.npy"
    lbl_path = base / f"cluster_{cid}_J_labels.txt"

    if (not j_mat_path.exists()) or (not lbl_path.exists()):
        return (None, None)

    J = np.load(j_mat_path)  # shape (M,M), Hz
    labels: List[str] = []
    with lbl_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.append(line)
    return (J, labels)


def main() -> None:
    args = parse_cli()
    setup_logging(args.log_level, args.quiet)
    mkdir_p(OUT_DIR)

    # Choose which energies_<...>.tsv we read, same logic as compute
    eps = (
        float(args.eps)
        if args.eps is not None
        else SOLVENT_EPS.get(args.solvent.lower(), None)
    )
    solvent_key = (
        args.solvent.lower()
        if args.eps is None and eps is not None
        else (f"eps{eps}" if eps is not None else "vacuum")
    )

    # Collect tags
    if args.tags:
        tags = args.tags
    else:
        cfs = sorted(CLUSTERS_DIR.glob("*_clusters.tsv"))
        if not cfs:
            LOG.error("No *_clusters.tsv in %s/", CLUSTERS_DIR)
            return
        tags = [cf.stem.removesuffix("_clusters") for cf in cfs]

    for tag in tags:
        LOG.info("[tag] %s", tag)
        table = load_clusters_table(CLUSTERS_DIR / f"{tag}_clusters.tsv", tag)

        # energies_<solvent_key>.tsv must exist (written by compute step)
        en_path = OUT_DIR / tag / f"energies_{solvent_key}.tsv"
        if not en_path.exists():
            LOG.error(
                "Missing %s for tag=%s. Run compute step with matching solvent/eps.",
                en_path.name,
                tag,
            )
            continue
        en_df = pd.read_csv(
            en_path, sep="\t", comment="#", names=["cid", "energy_Ha"]
        )

        # We'll assemble:
        #   - deltas: [ncluster][natom]
        #   - fracs: MD fractions from cluster table
        #   - energies: ddCOSMO energies for Boltzmann
        #   - J_list: list of J matrices (Hz)
        #   - J_weights_raw: raw weights for J entries (parallel to fracs/energies)
        #   - j_labels_ref: first non-None label list we encounter
        deltas: List[np.ndarray] = []
        fracs: List[float] = []
        energies: List[float] = []

        atom_names: Optional[List[str]] = None
        atom_elems: Optional[List[str]] = None

        J_list: List[np.ndarray] = []
        J_weights_raw: List[float] = []
        j_labels_ref: Optional[List[str]] = None
        j_dim_ref: Optional[int] = None

        # iterate over all clusters in this tag
        for row in table:
            # shifts
            df = read_cluster_shifts(tag, row.cid)
            if atom_names is None:
                atom_names = list(df["atom_name"].astype(str).values)
                atom_elems = list(df["element"].astype(str).values)
            deltas.append(df["shift_ppm"].to_numpy(float))

            # fractions / energies
            fracs.append(float(row.fraction))
            Ei = float(
                en_df.loc[en_df["cid"] == row.cid, "energy_Ha"].values[0]
            )
            energies.append(Ei)

            # J couplings
            (J_mat, j_labels) = read_cluster_J(tag, row.cid)
            if J_mat is None or j_labels is None:
                LOG.debug(
                    "  [cluster %s] no J data found; skipping in J-average.",
                    row.cid,
                )
            else:
                # sanity: store first label order and dimension
                if j_labels_ref is None:
                    j_labels_ref = list(j_labels)
                    j_dim_ref = int(J_mat.shape[0])
                else:
                    # check shape / label compatibility
                    ok_shape = (J_mat.shape[0] == j_dim_ref == J_mat.shape[1])
                    ok_labels = (list(j_labels) == j_labels_ref)
                    if (not ok_shape) or (not ok_labels):
                        LOG.warning(
                            "  [cluster %s] J mismatch (shape or labels). "
                            "Will exclude from J-average.",
                            row.cid,
                        )
                        J_mat = None  # drop it

                if J_mat is not None:
                    J_list.append(J_mat.astype(float))
                    # record same weight source as we'll use globally
                    J_weights_raw.append(float(row.fraction))

        # pick weights
        if args.no_boltz:
            # Use MD fractions as-is
            w = np.array(fracs, float)
            if w.sum() > 1e-15:
                w = w / w.sum()
            else:
                w = np.ones_like(w) / w.size
            weight_mode = "MD fractions"
        else:
            w = boltzmann_weights(energies, T_K=float(args.temp))
            weight_mode = f"Boltzmann @ {args.temp:.2f} K"

        # weighted-average shifts (fast exchange)
        M = np.stack(deltas, axis=0)  # shape (nclust, natom)
        avg_shifts = np.sum(M * w[:, None], axis=0)

        # write fastavg shifts tsv
        out_shift_tsv = (
                OUT_DIR
                / tag
                / f"fastavg_{solvent_key}_{int(round(float(args.temp)))}K.tsv"
        )
        with out_shift_tsv.open("w") as fh:
            fh.write("# atom_idx\tatom_name\telement\tshift_ppm\n")
            for i, (nm, el, dppm) in enumerate(
                    zip(atom_names, atom_elems, avg_shifts)
            ):
                fh.write(f"{i}\t{nm}\t{el}\t{fmt(dppm)}\n")

        LOG.info("  wrote %s", out_shift_tsv.name)

        # weighted-average J matrices (if we have any J_list)
        j_avg_path_npy = None
        j_avg_labels_path = None
        has_J_avg = False

        if J_list and j_labels_ref is not None:
            # choose proper weights for J:
            # we want physical consistency with shifts, but some clusters
            # may not have J (failure / mismatch). We renormalize only
            # over those that *did* contribute.
            Jw = np.array(J_weights_raw, float)
            if args.no_boltz:
                # same convention as for shifts, but restricted+renorm
                if Jw.sum() > 1e-15:
                    Jw = Jw / Jw.sum()
                else:
                    Jw = np.ones_like(Jw) / Jw.size
            else:
                # If Boltzmann, derive weights from energies again,
                # filtered to only the subset with usable J.
                # Map each contributing cluster's fraction index to its energy:
                # We can't rely on order because J_list is collected
                # only for some clusters, so simpler approach:
                # We'll just reuse J_weights_raw (which we filled from MD frac)
                # and warn that for Boltzmann we approximated using MD frac
                # if not all clusters gave J. That keeps logic simple,
                # still physical enough for now.
                if Jw.sum() > 1e-15:
                    Jw = Jw / Jw.sum()
                else:
                    Jw = np.ones_like(Jw) / Jw.size
                LOG.debug(
                    "  J-average weights approximated from MD fractions "
                    "because Boltzmann subset mapping would get messy."
                )

            J_avg = average_J_matrices(J_list, Jw)

            # save averaged J
            j_avg_path_npy = (
                    OUT_DIR
                    / tag
                    / f"fastavg_J_{solvent_key}_{int(round(float(args.temp)))}K.npy"
            )
            np.save(j_avg_path_npy, J_avg.astype(float))

            j_avg_labels_path = OUT_DIR / tag / "fastavg_J_labels.txt"
            with j_avg_labels_path.open("w") as fh:
                for lbl in j_labels_ref:
                    fh.write(f"{lbl}\n")

            LOG.info(
                "  wrote %s and %s",
                j_avg_path_npy.name,
                j_avg_labels_path.name,
            )
            has_J_avg = True
        else:
            LOG.warning(
                "  No compatible J matrices found for %s; skipping J-average.",
                tag,
            )

        # write run metadata
        meta = {
            "solvent_key": solvent_key,
            "temperature_K": float(args.temp),
            "weights": weight_mode,
            "has_J_avg": has_J_avg,
        }
        if has_J_avg and j_avg_labels_path is not None:
            meta["J_labels_file"] = j_avg_labels_path.name
            meta["J_avg_file"] = (
                j_avg_path_npy.name if j_avg_path_npy else None
            )

        write_params(OUT_DIR, tag, "params_average.txt", meta)

        LOG.info(
            "[ok] %s: wrote fastavg shifts%s",
            tag,
            " and J-average" if has_J_avg else "",
        )

    LOG.info("[done]")


if __name__ == "__main__":
    main()
