#!/usr/bin/env python3
# f_predict_shifts_compute.py
#
# Heavy step:
#   - for each conformer cluster:
#       * run DFT (PySCF) to get isotropic shielding σ_iso
#       * convert to chemical shifts δ (ppm) vs TMS ref
#       * compute scalar spin–spin J couplings (Hz)
#       * compute ddCOSMO single-point energy for Boltzmann weights
#       * write all cluster-level outputs
#
# Also writes:
#   - energies_<solvent_key>.tsv         (per-cluster energies in Ha)
#   - params_compute.txt                 (metadata for this tag/solvent)
#   - cluster_<cid>_shifts.tsv           (σ_iso and δ per atom)
#   - cluster_<cid>_j_couplings.tsv      (upper triangle J_ij in Hz)
#   - cluster_<cid>_J.npy                (full J matrix, Hz, np.save format)
#   - cluster_<cid>_J_labels.txt         (spin labels order matching J.npy)
#
# 2025-11-01

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Sequence

import numpy as np
import pyscf

from f_predict_shifts_core import (
    OUT_DIR,
    CLUSTERS_DIR,
    SOLVENT_EPS,
    setup_logging,
    detect_gpu,
    mkdir_p,
    get_charge_spin,
    load_clusters_table,
    read_pdb_atoms,
    compute_sigma_iso,
    compute_spinspin_JHz,
    sigma_to_delta,
    sp_energy_pcm,
    tms_ref_sigma,
    write_cluster_shifts,
    write_j_couplings,
    write_params,
)

LOG = logging.getLogger("nmrshifts.compute")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-cluster NMR shifts, scalar J couplings, and PCM energies."
    )
    p.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Tags to process (default: all *_clusters.tsv).",
    )
    p.add_argument("--xc", type=str, default="b3lyp", help="DFT functional.")
    p.add_argument("--basis", type=str, default="def2-tzvp", help="Basis set.")
    p.add_argument(
        "--gpu",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use gpu4pyscf if available.",
    )
    p.add_argument(
        "--solvent",
        type=str,
        default="DMSO",
        help="Solvent name for PCM weighting energies (e.g., DMSO, CDCl3).",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Override dielectric; if set, overrides --solvent.",
    )
    p.add_argument(
        "--keep-isotopes",
        type=str,
        nargs="*",
        default=["1H"],
        help='Which nuclei to include in J (default: 1H only). '
             'Add "13C" if you also want heteronuclear couplings for HSQC etc.',
    )
    p.add_argument(
        "--no-opt",
        action="store_true",
        help="(Reserved) Skip geometry opt (currently always skipped).",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR."
    )
    p.add_argument(
        "--quiet", action="store_true", help="Reduce informational messages."
    )
    return p.parse_args()


def _save_J_aux(out_dir, tag: str, cid: int, labels: Sequence[str], J_Hz: np.ndarray) -> None:
    """
    Persist machine-readable J matrix and its ordering for downstream steps.

    cluster_<cid>_J.npy:
        np.save() of full square matrix [M,M] in Hz
    cluster_<cid>_J_labels.txt:
        one label per line, order matches axes of J.npy
    """
    base = OUT_DIR / tag
    base.mkdir(parents=True, exist_ok=True)

    npy_path = base / f"cluster_{cid}_J.npy"
    np.save(npy_path, J_Hz.astype(float))

    lbl_path = base / f"cluster_{cid}_J_labels.txt"
    with lbl_path.open("w") as fh:
        for lbl in labels:
            fh.write(f"{lbl}\n")


def main() -> None:
    args = parse_cli()
    setup_logging(args.log_level, args.quiet)
    mkdir_p(OUT_DIR)

    # GPU / backend selection
    gpu_info = detect_gpu()
    use_gpu = (args.gpu == "on") or (args.gpu == "auto" and gpu_info["gpu4pyscf"])
    if use_gpu and not gpu_info["gpu4pyscf"]:
        LOG.warning("GPU requested but gpu4pyscf not available; running CPU.")
        use_gpu = False

    # solvent dielectric for weighting energies
    eps = (
        float(args.eps)
        if args.eps is not None
        else SOLVENT_EPS.get(args.solvent.lower(), None)
    )
    if args.eps is not None:
        solvent_key = f"eps{eps}"
    elif eps is not None:
        solvent_key = args.solvent.lower()
    else:
        solvent_key = "vacuum"

    LOG.info(
        "PySCF %s | DFT=%s | basis=%s | GPU=%s | PCM=%s",
        getattr(pyscf, "__version__", "unknown"),
        args.xc,
        args.basis,
        ("ON" if use_gpu else "OFF"),
        ("off" if eps is None else f"ddCOSMO eps={eps:.2f}"),
    )

    # Tag discovery
    if args.tags:
        tags = args.tags
    else:
        cfs = sorted(CLUSTERS_DIR.glob("*_clusters.tsv"))
        if not cfs:
            LOG.error("No *_clusters.tsv in %s/", CLUSTERS_DIR)
            return
        tags = [cf.stem.removesuffix("_clusters") for cf in cfs]

    # Process each tag independently
    for tag in tags:
        LOG.info("[tag] %s", tag)
        table = load_clusters_table(CLUSTERS_DIR / f"{tag}_clusters.tsv", tag)

        # Global charge/spin guess from tag (you can always override later if needed)
        (charge, spin) = get_charge_spin(tag)

        # Reference σ(TMS) so we can convert σ_iso → δ (ppm)
        ref_sigma = tms_ref_sigma(args.xc, args.basis)

        energies_Ha: List[float] = []
        first_labels: Optional[List[str]] = None

        # Per-cluster loop
        for row in table:
            atom_names, symbols, coords_A = read_pdb_atoms(row.rep_pdb)

            LOG.info("  [cluster %s] %s", row.cid, row.rep_pdb.name)

            # 1. Shieldings σ_iso (gas-phase calc) → shifts δ ppm
            sigma_iso = compute_sigma_iso(
                symbols,
                coords_A,
                charge,
                spin,
                args.xc,
                args.basis,
                use_gpu=use_gpu,
            )
            delta_ppm = sigma_to_delta(symbols, sigma_iso, ref_sigma)
            write_cluster_shifts(
                OUT_DIR, tag, row.cid, atom_names, symbols, sigma_iso, delta_ppm
            )

            # 2. Scalar spin–spin couplings J_ij in Hz (solution-state)
            try:
                (J_Hz, kept_labels) = compute_spinspin_JHz(
                    symbols,
                    coords_A,
                    charge,
                    spin,
                    args.xc,
                    args.basis,
                    use_gpu=use_gpu,
                    isotopes_keep=args.keep_isotopes,
                )

                if J_Hz.size:
                    # write human-readable triangular TSV
                    write_j_couplings(OUT_DIR, tag, row.cid, kept_labels, J_Hz)
                    # write machine-readable .npy plus label order
                    _save_J_aux(OUT_DIR, tag, row.cid, kept_labels, J_Hz)

                    # remember first label order to record in params
                    if first_labels is None:
                        first_labels = list(kept_labels)

                else:
                    LOG.warning(
                        "    [cluster %s] No nuclei kept for J (keep-isotopes=%s).",
                        row.cid,
                        args.keep_isotopes,
                    )

            except Exception as e:
                LOG.error(
                    "    [cluster %s] J coupling computation failed: %s",
                    row.cid,
                    e,
                )

            # 3. ddCOSMO single-point energy for Boltzmann weights
            e_pcm = sp_energy_pcm(
                symbols,
                coords_A,
                charge,
                spin,
                args.xc,
                args.basis,
                eps,
                use_gpu=use_gpu,
            )
            energies_Ha.append(float(e_pcm))

        # 4. Save energies_<solvent_key>.tsv (for weighting / Boltzmann later)
        energies_path = OUT_DIR / tag / f"energies_{solvent_key}.tsv"
        with energies_path.open("w") as fh:
            fh.write("# cid\tenergy_Ha\n")
            for r, E in zip(table, energies_Ha):
                fh.write(f"{r.cid}\t{E:.12f}\n")

        # 5. Save run metadata
        meta = {
            "xc": args.xc,
            "basis": args.basis,
            "charge_spin": (charge, spin),
            "gpu": use_gpu,
            "pcm_eps": (eps if eps is not None else "None"),
            "solvent_key": solvent_key,
            "notes": (
                "Per-cluster shieldings (gas-phase), scalar J couplings (Hz) "
                "and ddCOSMO single-point energies (Hartree) for Boltzmann weighting."
            ),
            "keep_isotopes": args.keep_isotopes,
        }
        if first_labels is not None:
            meta["spin_label_order"] = first_labels

        write_params(OUT_DIR, tag, "params_compute.txt", meta)

        LOG.info(
            "[ok] %s: wrote cluster shifts, J couplings, and %s.",
            tag,
            energies_path.name,
        )

    LOG.info("[done]")


if __name__ == "__main__":
    main()
