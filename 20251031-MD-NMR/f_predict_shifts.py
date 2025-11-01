#!/usr/bin/env python3
# f_predict_shifts.py
#
# Main entrypoint / dispatcher for the NMR observable prediction pipeline.
#
# Subcommands:
#   compute  -> heavy per-cluster quantum chemistry (shieldings → shifts,
#               scalar J couplings, ddCOSMO energies)
#   average  -> fast Boltzmann/MD averaging across clusters (shifts, J)
#
# 2025-11-01

"""
f_predict_shifts.py
===================

This script provides two pipeline stages:

1) compute  (heavy, per-cluster DFT step)
   • For each tag (system), read clustered conformer representatives from:
       e_cluster/<tag>_clusters.tsv
       e_cluster/<tag>_cluster_<cid>_rep.pdb
   • For each cluster representative, run (U)KS DFT with PySCF to obtain:
       - isotropic nuclear shieldings σ_iso
       - convert σ_iso → chemical shifts δ (ppm) vs TMS
       - scalar spin–spin couplings J_ij (Hz) via PySCF SSC
       - ddCOSMO single-point energy (Hartree) for Boltzmann weighting
   • Write per-cluster outputs under f_predict_shifts/<tag>/ :
       cluster_<cid>_shifts.tsv
         atom_idx, atom_name, element, sigma_iso, shift_ppm
       cluster_<cid>_j_couplings.tsv
         i, j, label_i, label_j, J_Hz (upper triangle only)
       cluster_<cid>_J.npy
         full J matrix [M,M] in Hz
       cluster_<cid>_J_labels.txt
         nucleus labels / ordering for that J matrix
   • Also write per-tag:
       energies_<solventkey>.tsv
         cid, energy_Ha (PCM/ddCOSMO single-point energies)
       params_compute.txt
         xc/basis/GPU/PCM metadata, and keep_isotopes used for J

   Notes:
   - Shieldings σ_iso are currently evaluated in gas phase for consistency.
   - PCM (ddCOSMO) is only used for single-point energies (population weighting).
   - gpu4pyscf can accelerate SCF. Property evaluation (NMR / SSC) may fall
     back to CPU internally.

2) average  (fast, no new DFT)
   • Read for each tag:
       f_predict_shifts/<tag>/cluster_<cid>_shifts.tsv
       f_predict_shifts/<tag>/energies_<solventkey>.tsv
       f_predict_shifts/<tag>/cluster_<cid>_J.npy (if present)
   • Compute weights per cluster:
       - Boltzmann weights at temperature T using ddCOSMO energies, OR
       - direct MD fractions from the cluster table (--no-boltz)
   • Produce fast-exchange weighted averages:
       fastavg_<solventkey>_<T>K.tsv
         atom_idx, atom_name, element, shift_ppm  (⟨δ⟩)
       fastavg_J_<solventkey>_<T>K.npy
         ⟨J⟩ matrix in Hz (elementwise weighted average of J matrices)
       fastavg_J_labels.txt
         nucleus ordering for that ⟨J⟩
       params_average.txt
         temperature, solvent key, which weighting was used, etc.

   This step is cheap: it does math and file I/O only, so you can re-run it
   quickly for different temperatures / solvents / dielectric constants
   without re-running "compute".

File layout (per tag)
---------------------
f_predict_shifts/<tag>/:
  cluster_<cid>_shifts.tsv
  cluster_<cid>_j_couplings.tsv
  cluster_<cid>_J.npy
  cluster_<cid>_J_labels.txt
  energies_<solventkey>.tsv
  fastavg_<solventkey>_<T>K.tsv
  fastavg_J_<solventkey>_<T>K.npy
  fastavg_J_labels.txt
  params_compute.txt
  params_average.txt

<solventkey> is either:
  - the lowercase solvent name you passed (e.g. "dmso", "cdcl3"), or
  - "eps<value>" if you forced a dielectric with --eps

Typical usage
-------------
(1) Heavy per-cluster quantum step:

  ./f_predict_shifts.py compute -- --tags aspirin_neutral_cdcl3 \
      --xc b3lyp --basis def2-tzvp --solvent DMSO --gpu auto \
      --keep-isotopes 1H

This writes shifts, J couplings, and energies for each cluster of that tag.

(2) Average across clusters at chosen temperature / solvent:

  ./f_predict_shifts.py average -- --tags aspirin_neutral_cdcl3 \
      --solvent DMSO --temp 298.15

This writes fastavg_<solvent>_<temp>K.tsv and fastavg_J_<solvent>_<temp>K.npy.

CLI summary for this dispatcher
-------------------------------

compute:
  --tags TAG [TAG ...]
  --xc XC
  --basis BASIS
  --gpu {auto,on,off}
  --solvent NAME
  --eps EPS
  --keep-isotopes ISO [ISO ...]     nuclei to include in J (e.g. 1H 13C)
  --log-level LEVEL
  --quiet

average:
  --tags TAG [TAG ...]
  --temp K
  --solvent NAME
  --eps EPS
  --no-boltz
  --log-level LEVEL
  --quiet

Implementation details
----------------------
This wrapper dispatches to:
    f_predict_shifts_compute.main()
    f_predict_shifts_average.main()

All argument parsing for each sub-step lives in those modules.

An optional `--` after the subcommand is supported for readability:
    ./f_predict_shifts.py compute -- --tags foo
is treated the same as:
    ./f_predict_shifts.py compute --tags foo

No PySCF or other heavy deps are imported here; they live in the sub-scripts.
"""

from __future__ import annotations

import argparse
import sys


def _strip_separators(argv: list[str]) -> list[str]:
    """
    Remove one leading '--' if present, and also remove a standalone '--'
    immediately after the subcommand.

    Examples:
      ["compute","--","--tags","foo"] -> ["compute","--tags","foo"]
      ["--","compute","--","--tags","foo"] -> ["compute","--tags","foo"]
    """
    out = list(argv)
    # drop leading --
    if out and out[0] == "--":
        out = out[1:]
    # if we have at least [cmd, "--", ...] collapse that
    if len(out) >= 2 and out[1] == "--":
        out = [out[0]] + out[2:]
    return out


def main() -> None:
    PROG = "f_predict_shifts.py"

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "NMR shift / J-coupling pipeline:\n"
            "  compute  → per-cluster DFT (σ→δ, J, energies)\n"
            "  average  → Boltzmann/MD fast averaging (δ, J)\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    # expose help text for subcommands
    sub.add_parser(
        "compute",
        help="Compute per-cluster shieldings→shifts, J couplings, and PCM energies.",
    )
    sub.add_parser(
        "average",
        help="Average precomputed shifts and J with Boltzmann/MD fractions.",
    )

    if len(sys.argv) == 1:
        # Just print high-level help plus quick examples.
        parser.print_help(sys.stderr)
        print(
            r"""
Examples:

  # 1) Heavy step: per-cluster NMR, J, and PCM energies (GPU if available)
  ./f_predict_shifts.py compute -- --tags aspirin_neutral_cdcl3 \
      --xc b3lyp --basis def2-tzvp --solvent DMSO --gpu auto \
      --keep-isotopes 1H

  # 2) Fast step: average at chosen temperature / solvent
  ./f_predict_shifts.py average -- --tags aspirin_neutral_cdcl3 \
      --solvent DMSO --temp 298.15
"""
        )
        return

    # normalize argv:
    argv = _strip_separators(sys.argv[1:])
    if not argv:
        parser.print_help(sys.stderr)
        return

    # first token should be the subcommand
    cmd = argv[0]
    rest = argv[1:]

    # in case user did "<cmd> -- --tags ...", strip again
    rest = _strip_separators([cmd] + rest)[1:]

    dispatch = {
        "compute": ("f_predict_shifts_compute", "f_predict_shifts_compute.py"),
        "average": ("f_predict_shifts_average", "f_predict_shifts_average.py"),
    }

    if cmd not in dispatch:
        parser.print_help(sys.stderr)
        print(
            f"\n[error] Unknown subcommand: {cmd!r}. "
            "Use 'compute' or 'average'."
        )
        return

    (module_name, child_prog) = dispatch[cmd]

    # Import lazily and hand off to the real main()
    module = __import__(module_name, fromlist=["main"])
    sys.argv = [child_prog] + rest
    return module.main()


if __name__ == "__main__":
    main()
