#!/usr/bin/env python3
# f_predict_shifts_average.py

# Average (fast step): Boltzmann or MD-fraction weights → fast-exchange shifts
# 2025-11-01

from __future__ import annotations

import argparse
import logging
from typing import List

import numpy as np
import pandas as pd

from f_predict_shifts_core import (
    OUT_DIR, CLUSTERS_DIR, SOLVENT_EPS,
    setup_logging, mkdir_p,
    load_clusters_table,
    boltzmann_weights, fmt, write_params,
)

LOG = logging.getLogger("nmrshifts.average")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Average precomputed per-cluster shifts with temperature/solvent weights."
    )
    p.add_argument("--tags", type=str, nargs="*", help="Tags to process (default: all *_clusters.tsv).")
    p.add_argument("--temp", type=float, default=298.15, help="Temperature [K] for Boltzmann weights.")
    p.add_argument("--solvent", type=str, default="DMSO", help="Solvent name to pick energies_<solvent>.tsv.")
    p.add_argument("--eps", type=float, default=None, help="Override dielectric; uses energies_eps<val>.tsv.")
    p.add_argument("--no-boltz", action="store_true", help="Use MD fractions instead of Boltzmann weights.")
    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR.")
    p.add_argument("--quiet", action="store_true", help="Reduce informational messages.")
    return p.parse_args()


def read_cluster_shifts(tag: str, cid: int) -> pd.DataFrame:
    path = OUT_DIR / tag / f"cluster_{cid}_shifts.tsv"
    df = pd.read_csv(path, sep="\t", comment="#")
    return df  # columns: atom_idx, atom_name, element, sigma_iso, shift_ppm


def main() -> None:
    args = parse_cli()
    setup_logging(args.log_level, args.quiet)
    mkdir_p(OUT_DIR)

    # energies file key
    eps = float(args.eps) if args.eps is not None else SOLVENT_EPS.get(args.solvent.lower(), None)
    solvent_key = (
        args.solvent.lower() if args.eps is None and eps is not None else f"eps{eps}" if eps is not None else "vacuum")

    # tags
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

        # read energies for this solvent_key
        en_path = OUT_DIR / tag / f"energies_{solvent_key}.tsv"
        if not en_path.exists():
            LOG.error("Missing %s for tag=%s. Run: f_predict_shifts_compute.py --solvent %s", en_path.name, tag,
                      args.solvent)
            continue
        en_df = pd.read_csv(en_path, sep="\t", comment="#", names=["cid", "energy_Ha"])

        # gather per-cluster delta arrays (aligned by cid)
        deltas: List[np.ndarray] = []
        fracs: List[float] = []
        energies: List[float] = []
        atom_names = None
        atom_elems = None

        # we assume same atom ordering across clusters (reps)
        for row in table:
            df = read_cluster_shifts(tag, row.cid)
            if atom_names is None:
                atom_names = list(df["atom_name"].astype(str).values)
                atom_elems = list(df["element"].astype(str).values)
            deltas.append(df["shift_ppm"].to_numpy(float))
            fracs.append(float(row.fraction))
            Ei = float(en_df.loc[en_df["cid"] == row.cid, "energy_Ha"].values[0])
            energies.append(Ei)

        if args.no_boltz:
            w = np.array(fracs, float)
            w = w / w.sum() if w.sum() > 1e-15 else np.ones_like(w) / w.size
            weight_mode = "MD fractions"
        else:
            w = boltzmann_weights(energies, T_K=float(args.temp))
            weight_mode = f"Boltzmann @ {args.temp:.2f} K"

        M = np.stack(deltas, axis=0)  # (nclust, natom)
        avg = np.sum(M * w[:, None], axis=0)

        # write fast average
        out = OUT_DIR / tag / f"fastavg_{solvent_key}_{int(round(float(args.temp)))}K.tsv"
        with out.open("w") as fh:
            fh.write("# atom_idx\tatom_name\telement\tshift_ppm\n")
            for i, (nm, el, dppm) in enumerate(zip(atom_names, atom_elems, avg)):
                fh.write(f"{i}\t{nm}\t{el}\t{fmt(dppm)}\n")

        write_params(
            OUT_DIR, tag, "params_average.txt",
            {
                "solvent_key": solvent_key,
                "temperature_K": float(args.temp),
                "weights": weight_mode,
            },
        )
        LOG.info("[ok] %s: wrote %s", tag, out.name)

    LOG.info("[done]")


if __name__ == "__main__":
    main()
