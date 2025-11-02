#!/usr/bin/env python3
# f_predict_shifts_compute.py
#
# Heavy step:
#   - for each conformer cluster:
#       * SCF (DFT) at given xc/basis (GPU if allowed for SCF only)
#       * isotropic shielding σ_iso → chemical shifts δ (ppm) vs TMS
#       * scalar spin–spin J couplings (Hz) from SSC, if available
#       * ddCOSMO single-point energy (Hartree) for Boltzmann weights
#
# Per-cluster outputs (in f_predict_shifts/<tag>/):
#   - cluster_<cid>_shifts.tsv
#       atom_idx, atom_name, element, σ_iso, δ(ppm)
#   - cluster_<cid>_j_couplings.tsv         [if J available]
#       i, j, label_i, label_j, J_Hz (upper triangle)
#   - cluster_<cid>_J.npy                    [if J available]
#       dense (M,M) J matrix in Hz
#   - cluster_<cid>_J_labels.txt             [if J available]
#       nucleus labels in the J matrix order
#
# Per-tag outputs:
#   - energies_<solvent_key>.tsv
#       cid, ddCOSMO single-point energy (Hartree)
#   - params_compute.txt
#       method/basis/etc., and ssc_available=0/1
#
# Behavior of --require-j:
#   We now do a robust SSC/J preflight (assert_ssc_available_fast):
#     - build tiny H2, run SCF on CPU,
#     - call SSC, reduce pair tensors → scalar J,
#     - verify finite square J matrix.
#   If that fails:
#     * with --require-j → abort before any expensive work
#     * without --require-j → continue but skip J outputs
#
# This file assumes f_predict_shifts_core provides:
#   - compute_sigma_iso(), sigma_to_delta(), tms_ref_sigma()
#   - compute_spinspin_JHz()  (wraps SSC and builds J matrix)
#   - sp_energy_pcm()
#   - assert_ssc_available_fast()  (updated to be tolerant of PySCF formats)
#   - write_cluster_shifts(), write_j_couplings(), write_params()
#   - load_clusters_table(), read_pdb_atoms(), get_charge_spin()
#   - SOLVENT_EPS map, etc.
#
# :contentReference[oaicite:0]{index=0}


from __future__ import annotations

import argparse
import logging
from typing import List, Optional

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
    sigma_to_delta,
    tms_ref_sigma,
    write_cluster_shifts,
    write_j_couplings,
    write_params,
    assert_ssc_available_fast,
    compute_sigma_J_and_energy_once,
)

LOG = logging.getLogger("nmrshifts.compute")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-cluster quantum step: chemical shifts δ (ppm), "
            "scalar J couplings (Hz, if available), and ddCOSMO energies."
        )
    )

    p.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Tags to process (default: infer from *_clusters.tsv).",
    )

    p.add_argument("--xc", type=str, default="b3lyp", help="DFT functional.")
    p.add_argument("--basis", type=str, default="def2-tzvp", help="Basis set.")

    p.add_argument(
        "--gpu",
        choices=["auto", "on", "off"],
        default="auto",
        help="Try gpu4pyscf for SCF (properties still run on CPU).",
    )

    p.add_argument(
        "--solvent",
        type=str,
        default="DMSO",
        help="Solvent label for ddCOSMO weighting (e.g. DMSO, CDCl3).",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=None,
        help=(
            "Override dielectric constant for ddCOSMO. "
            "If set, overrides --solvent."
        ),
    )

    p.add_argument(
        "--keep-isotopes",
        type=str,
        nargs="*",
        default=["1H"],
        help=(
            'Which nuclei to include in J (default: 1H). '
            'Add "13C" etc. if you want heteronuclear couplings.'
        ),
    )

    p.add_argument(
        "--require-j",
        action="store_true",
        help=(
            "Abort early if SSC/J is not usable. "
            "Without this flag, we still run shifts/energies and just skip J."
        ),
    )

    p.add_argument(
        "--no-opt",
        action="store_true",
        help="Reserved (geometry opt currently not performed).",
    )

    p.add_argument(
        "--tms-opt",
        action="store_true",
        help=(
            "Allow Berny geometry optimization for the TMS reference "
            "before computing σ_ref. "
            "By default (flag absent), TMS is NOT optimized; "
            "a symmetric guess geometry is used."
        ),
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
        help="Reduce informational logging.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_cli()
    setup_logging(args.log_level, args.quiet)
    mkdir_p(OUT_DIR)

    # GPU policy for SCF
    gpu_info = detect_gpu()
    use_gpu = (args.gpu == "on") or (args.gpu == "auto" and gpu_info["gpu4pyscf"])
    if use_gpu and not gpu_info["gpu4pyscf"]:
        LOG.warning("GPU requested but gpu4pyscf not available; forcing CPU.")
        use_gpu = False

    # dielectric and solvent key for output filenames
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

    # ------------------------------------------------------------------
    # Robust SSC/J preflight.
    # If this fails:
    #   --require-j → abort before heavy work
    #   otherwise   → run without J
    # ------------------------------------------------------------------
    try:
        assert_ssc_available_fast(
            xc=args.xc,
            basis=args.basis,
            isotopes_keep=args.keep_isotopes,
        )
        j_ok = True
    except Exception as e:
        j_ok = False
        if args.require_j:
            LOG.error(
                "Spin–spin coupling (SSC/J) preflight failed: %s",
                e,
            )
            LOG.error("--require-j was set → aborting before heavy SCF.")
            return
        LOG.warning(
            "J couplings will be skipped (SSC preflight failed: %s).",
            e,
        )

    # Discover tags if the user didn't pass any
    if args.tags:
        tags = args.tags
    else:
        cfs = sorted(CLUSTERS_DIR.glob("*_clusters.tsv"))
        if not cfs:
            LOG.error("No *_clusters.tsv in %s/", CLUSTERS_DIR)
            return
        tags = [cf.stem.removesuffix("_clusters") for cf in cfs]

    # TMS reference σ_ref so we can report δ(ppm).
    # We pass do_opt=... which controls Berny geometry relaxation
    # of the TMS reference only. Default (False) = skip Berny.
    ref_sigma = tms_ref_sigma(
        args.xc,
        args.basis,
        do_opt=args.tms_opt,
    )

    # Process each tag
    for tag in tags:
        LOG.info("[tag] %s", tag)
        table = load_clusters_table(CLUSTERS_DIR / f"{tag}_clusters.tsv", tag)

        # crude global charge, multiplicity guess from tag name
        (charge, spin) = get_charge_spin(tag)

        energies_Ha: List[float] = []
        first_labels: Optional[List[str]] = None

        # per-cluster loop
        for row in table:
            (atom_names, symbols, coords_A) = read_pdb_atoms(row.rep_pdb)

            LOG.info("  [cluster %s] %s", row.cid, row.rep_pdb.name)

            # --- SINGLE HEAVY CALL ---
            (
                sigma_iso,
                J_Hz,
                kept_labels,
                e_tot,
            ) = compute_sigma_J_and_energy_once(
                symbols=symbols,
                coords_A=coords_A,
                charge=charge,
                spin=spin,
                xc=args.xc,
                basis=args.basis,
                use_gpu=use_gpu,
                isotopes_keep=args.keep_isotopes,
                need_J=j_ok,
                eps=eps,
            )

            # σ_iso → δ(ppm) and write per-cluster shifts
            delta_ppm = sigma_to_delta(symbols, sigma_iso, ref_sigma)
            write_cluster_shifts(
                OUT_DIR,
                tag,
                row.cid,
                atom_names,
                symbols,
                sigma_iso,
                delta_ppm,
            )

            # Write J couplings only if j_ok and we actually have any
            if j_ok and J_Hz.size:
                write_j_couplings(
                    OUT_DIR,
                    tag,
                    row.cid,
                    kept_labels,
                    J_Hz,
                )
                if first_labels is None:
                    first_labels = list(kept_labels)
            elif j_ok:
                LOG.warning(
                    "    [cluster %s] No nuclei kept for J "
                    "(keep-isotopes=%s).",
                    row.cid,
                    args.keep_isotopes,
                )
            else:
                LOG.debug(
                    "    [cluster %s] Skipping J couplings (SSC disabled).",
                    row.cid,
                )

            # record energy (already PCM if eps != None)
            energies_Ha.append(float(e_tot))

        # Write energies_<solvent_key>.tsv
        energies_path = OUT_DIR / tag / f"energies_{solvent_key}.tsv"
        energies_path.parent.mkdir(parents=True, exist_ok=True)
        with energies_path.open("w") as fh:
            fh.write("# cid\tenergy_Ha\n")
            for (r, E) in zip(table, energies_Ha):
                fh.write(f"{r.cid}\t{E:.12f}\n")

        # Write metadata
        meta = {
            "xc": args.xc,
            "basis": args.basis,
            "charge_spin": (charge, spin),
            "gpu": use_gpu,
            "pcm_eps": (eps if eps is not None else "None"),
            "solvent_key": solvent_key,
            "keep_isotopes": args.keep_isotopes,
            "ssc_available": int(j_ok),
            "tms_opt": bool(args.tms_opt),
            "notes": (
                "Per-cluster shieldings σ_iso and δ(ppm), scalar J couplings "
                "(Hz, if ssc_available=1), and ddCOSMO single-point "
                "energies (Hartree) for Boltzmann weighting. "
                "All from ONE SCF per cluster. "
                "tms_opt=True means Berny optimization was allowed for the "
                "TMS reference used for δ(ppm)."
            ),
        }
        if first_labels is not None:
            meta["spin_label_order"] = first_labels

        write_params(OUT_DIR, tag, "params_compute.txt", meta)

        LOG.info(
            "[ok] %s: wrote cluster shifts%s and %s.",
            tag,
            (", J couplings" if j_ok else " (no J)"),
            energies_path.name,
        )

    LOG.info("[done]")


if __name__ == "__main__":
    main()
