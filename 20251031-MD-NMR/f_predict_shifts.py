#!/usr/bin/env python3
# Main entrypoint: subcommands that call the two drivers
# 2025-11-01

"""
f_predict_shifts.py
===================

Main entrypoint for a two-stage NMR shift pipeline with subcommands:

  1) compute — Heavy step:
     • For each tag (system), read clustered representatives from e_cluster/<tag>_clusters.tsv
     • Compute per-cluster *gas-phase* NMR shieldings → per-cluster shifts (TMS referenced)
     • Compute PCM single-point energies (for a chosen solvent or epsilon) for Boltzmann weights
     • Writes results under: f_predict_shifts/<tag>/

  2) average — Fast step:
     • Reweights precomputed per-cluster shifts with either Boltzmann weights (E_pcm, T)
       or MD fractions from the cluster table
     • Produces fast-exchange averaged shifts for any temperature and solvent/epsilon
     • Runs quickly: no PySCF recomputation required

Why split in two?
-----------------
NMR shieldings (gas-phase) are independent of temperature and typically independent
of PCM in this workflow. The heavy quantum step (“compute”) can be done once and
reused. Changing *temperature* or *solvent* only affects the weights, so the
“average” step is lightweight and fast—ideal for scanning conditions.

File layout
-----------
Outputs are organized per tag under f_predict_shifts/<tag>/:

  cluster_<cid>_shifts.tsv          # per-cluster per-atom shieldings & shifts (gas-phase)
  energies_<solventkey>.tsv         # per-cluster PCM single-point energies (Hartree)
  fastavg_<solventkey>_<T>K.tsv     # fast-exchange averaged shifts at T (written by "average")
  params_compute.txt                # record of compute settings (xc/basis/PCM/GPU etc.)
  params_average.txt                # record of averaging settings (temperature/weights)

The <solventkey> is either a known solvent name (e.g., "dmso", "cdcl3") or "eps<value>"
when an explicit dielectric is provided.

Basic usage
-----------
(1) Heavy step: compute per-cluster shifts and PCM energies

  ./f_predict_shifts.py compute -- --tags aspirin_neutral_cdcl3 \
      --xc b3lyp --basis def2-tzvp --solvent DMSO --gpu auto --log-level INFO

Notes:
  • Pass flags to the subcommand after a `--`. The main wrapper forwards them.
  • The compute step:
      - NMR shieldings/shifts are evaluated in gas phase (for consistency).
      - PCM ddCOSMO is used for *energies only* (weighting).
      - If gpu4pyscf is installed and `--gpu on/auto` is set, SCF uses the GPU backend
        (NMR property evaluation itself may be CPU-bound in many PySCF builds).

(2) Fast step: average at any temperature and solvent

  ./f_predict_shifts.py average -- --tags aspirin_neutral_cdcl3 \
      --solvent DMSO --temp 298.15

You can rerun “average” as often as you like to change:
  • Temperature:   --temp 273.15
  • Solvent name:  --solvent CDCl3
  • Dielectric:    --eps 20.7  (uses energies_eps20.7.tsv)
  • Weights mode:  --no-boltz  (use MD fractions from the cluster table)

Required inputs
---------------
Place the cluster table and representative PDBs in e_cluster/ with names like:
  e_cluster/<tag>_clusters.tsv
  e_cluster/<tag>_cluster_<cid>_rep.pdb

The TSV may include columns such as:
  cid, fraction (or weight/pop), rep_path (optional)
If fractions are missing, equal weights are assumed (unless Boltzmann is used in "average").

CLI summary (forwarded to sub-scripts)
--------------------------------------
compute:
  --tags TAG [TAG ...]     Tags to process (default: all *_clusters.tsv)
  --xc XC                  DFT functional (default: b3lyp)
  --basis BASIS            Basis set (default: def2-tzvp)
  --gpu {auto,on,off}      GPU SCF via gpu4pyscf if available (default: auto)
  --solvent NAME           Solvent for PCM energies (e.g., DMSO, CDCl3)
  --eps EPS                Override dielectric; if set, overrides --solvent
  --log-level LEVEL        DEBUG, INFO, WARNING, ERROR
  --quiet                  Reduce informational messages

average:
  --tags TAG [TAG ...]     Tags to process (default: all *_clusters.tsv)
  --temp K                 Temperature in Kelvin for Boltzmann weights (default: 298.15)
  --solvent NAME           Select energies_<solvent>.tsv for weighting
  --eps EPS                Select energies_eps<EPS>.tsv instead of a solvent name
  --no-boltz               Use MD fractions instead of Boltzmann weights
  --log-level LEVEL        DEBUG, INFO, WARNING, ERROR
  --quiet                  Reduce informational messages

Troubleshooting
---------------
• If GPU usage appears low:
    - The SCF can run on GPU, but NMR response/PCM steps may still be CPU-bound.
    - Ensure `gpu4pyscf` matches your PySCF/CUDA/CuPy versions; use `--gpu on`.

• If a representative PDB cannot be found:
    - The code expects e_cluster/<tag>_cluster_<cid>_rep.pdb (with a few fallbacks).
    - Check filenames or provide a 'rep_path' column in the cluster TSV.

This wrapper only dispatches to the sub-scripts. See
`f_predict_shifts_compute.py` and `f_predict_shifts_average.py` for full flag details.
"""

from __future__ import annotations


def main():
    import argparse, sys

    PROG = "f_predict_shifts.py"
    p = argparse.ArgumentParser(
        prog=PROG,
        description="NMR shifts pipeline (compute heavy step, then average reweighting).",
        add_help=True,
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("compute", help="Compute per-cluster shifts and PCM energies.")
    sub.add_parser("average", help="Average precomputed shifts with Boltzmann/MD fractions.")

    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        print(r"""
Examples:

  # 1) Heavy step: per-cluster NMR & PCM energies (GPU on if available)
  ./f_predict_shifts.py compute --tags aspirin_neutral_cdcl3 --xc b3lyp --basis def2-tzvp --solvent DMSO --gpu auto

  # 1a) Same as above (allows a separator after the subcommand too)
  ./f_predict_shifts.py compute -- --tags aspirin_neutral_cdcl3 --xc b3lyp --basis def2-tzvp --solvent DMSO --gpu auto

  # 2) Fast step: average at chosen temperature and solvent
  ./f_predict_shifts.py average --tags aspirin_neutral_cdcl3 --solvent DMSO --temp 298.15

Notes:
  - Flags after the subcommand are forwarded to the specific script.
  - A lone '--' separator is optional; both styles work.
  - Outputs live under f_predict_shifts/<tag>/ :
      cluster_<cid>_shifts.tsv
      energies_<solvent>.tsv
      fastavg_<solvent>_<T>K.tsv
""")
        return

    # Accept a stray top-level '--' (e.g., "script.py -- compute ...")
    argv_all = sys.argv[1:]
    if argv_all and argv_all[0] == "--":
        argv_all = argv_all[1:]

    if not argv_all:
        p.print_help(sys.stderr)
        return

    cmd = argv_all[0]
    rest = argv_all[1:]

    # Also accept a separator immediately after the subcommand
    if rest and rest[0] == "--":
        rest = rest[1:]

    if cmd == "compute":
        from f_predict_shifts_compute import main as run_compute
        sys.argv = ["f_predict_shifts_compute.py"] + rest
        return run_compute()

    if cmd == "average":
        from f_predict_shifts_average import main as run_average
        sys.argv = ["f_predict_shifts_average.py"] + rest
        return run_average()

    # Unknown subcommand → show help and a hint
    p.print_help(sys.stderr)
    print(f"\n[error] Unknown subcommand: {cmd!r}. Use 'compute' or 'average'.")


if __name__ == "__main__":
    main()
