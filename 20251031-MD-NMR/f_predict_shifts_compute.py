#!/usr/bin/env python3
# f_predict_shifts_compute.py

# Compute per-cluster NMR shifts and PCM energies (heavy step)
# 2025-11-01

from __future__ import annotations

import argparse
import logging
from typing import List

import pyscf

from f_predict_shifts_core import (
    OUT_DIR, CLUSTERS_DIR, SOLVENT_EPS,
    setup_logging, detect_gpu, mkdir_p,
    get_charge_spin, load_clusters_table, read_pdb_atoms,
    compute_sigma_iso, sigma_to_delta, sp_energy_pcm, tms_ref_sigma,
    write_cluster_shifts, write_params,
)

LOG = logging.getLogger("nmrshifts.compute")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-cluster NMR shifts and PCM energies."
    )
    p.add_argument("--tags", type=str, nargs="*", help="Tags to process (default: all *_clusters.tsv).")
    p.add_argument("--xc", type=str, default="b3lyp", help="DFT functional.")
    p.add_argument("--basis", type=str, default="def2-tzvp", help="Basis set.")
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="Use gpu4pyscf if available.")
    p.add_argument("--solvent", type=str, default="DMSO", help="Solvent name for PCM energies (e.g., DMSO, CDCl3).")
    p.add_argument("--eps", type=float, default=None, help="Override dielectric; if set, overrides --solvent.")
    p.add_argument("--no-opt", action="store_true", help="(Reserved) Skip geometry opt (currently always skipped).")
    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR.")
    p.add_argument("--quiet", action="store_true", help="Reduce informational messages.")
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    setup_logging(args.log_level, args.quiet)
    mkdir_p(OUT_DIR)

    # GPU banner
    gpu_info = detect_gpu()
    use_gpu = (args.gpu == "on") or (args.gpu == "auto" and gpu_info["gpu4pyscf"])
    if use_gpu and not gpu_info["gpu4pyscf"]:
        LOG.warning("GPU requested but gpu4pyscf not available; running CPU.")
        use_gpu = False

    # solvent
    eps = float(args.eps) if args.eps is not None else SOLVENT_EPS.get(args.solvent.lower(), None)
    solvent_key = (
        args.solvent.lower() if args.eps is None and eps is not None else f"eps{eps}" if eps is not None else "vacuum")
    LOG.info("PySCF %s | DFT=%s | basis=%s | GPU=%s | PCM=%s",
             getattr(pyscf, "__version__", "unknown"),
             args.xc, args.basis,
             ("ON" if use_gpu else "OFF"),
             ("off" if eps is None else f"ddCOSMO eps={eps:.2f}"))

    # Tags
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
        charge, spin = get_charge_spin(tag)
        ref = tms_ref_sigma(args.xc, args.basis)

        # outputs for energies
        energies_Ha: List[float] = []

        for row in table:
            atom_names, symbols, coords_A = read_pdb_atoms(row.rep_pdb)

            # NMR shieldings (gas-phase) → delta (TMS ref)
            sigma = compute_sigma_iso(symbols, coords_A, charge, spin, args.xc, args.basis, use_gpu=use_gpu)
            delta = sigma_to_delta(symbols, sigma, ref)
            write_cluster_shifts(OUT_DIR, tag, row.cid, atom_names, symbols, sigma, delta)

            # PCM single-point energies for weighting
            e_pcm = sp_energy_pcm(symbols, coords_A, charge, spin, args.xc, args.basis, eps)
            energies_Ha.append(float(e_pcm))

        # Save energies for this solvent key
        energies_path = OUT_DIR / tag / f"energies_{solvent_key}.tsv"
        with energies_path.open("w") as fh:
            fh.write("# cid\tenergy_Ha\n")
            for r, E in zip(table, energies_Ha):
                fh.write(f"{r.cid}\t{E:.12f}\n")

        write_params(
            OUT_DIR, tag, "params_compute.txt",
            {
                "xc": args.xc, "basis": args.basis,
                "charge_spin": (charge, spin),
                "gpu": use_gpu,
                "pcm_eps": (eps if eps is not None else "None"),
                "solvent_key": solvent_key,
                "notes": "Per-cluster shieldings (gas-phase) & PCM energies for weighting.",
            },
        )
        LOG.info("[ok] %s: wrote cluster shifts and %s.", tag, energies_path.name)

    LOG.info("[done]")


if __name__ == "__main__":
    main()
