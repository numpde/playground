#!/usr/bin/env python3
# f_predict_shifts.py
#
# Entry point / dispatcher for the NMR observable prediction pipeline.
#
# Subcommands:
#   compute  → heavy per-cluster quantum chemistry
#               - shielding tensors → chemical shifts
#               - optional scalar J couplings
#               - ddCOSMO single-point energies
#
#   average  → fast post-processing / ensemble averaging
#               - Boltzmann (or MD) weights over clusters
#               - weighted average shifts and J
#
# This file itself stays lightweight: it does not import PySCF.
# Each subcommand owns its own argparse and heavy deps.
#
# Usage pattern:
#   ./f_predict_shifts.py compute -- [compute args...]
#   ./f_predict_shifts.py average -- [average args...]
#
# The extra "--" is optional; we strip it for you.
#
# The compute subcommand supports --require-j (abort if SSC/J is not usable)
# and --keep-isotopes (e.g. 1H, 13C). The average subcommand does not care.


from __future__ import annotations

import argparse
import sys


def _strip_separators(argv: list[str]) -> list[str]:
    """
    Remove cosmetic standalone '--' that users may insert after the subcommand
    for readability.

    Examples:
      ["compute","--","--tags","foo"] -> ["compute","--tags","foo"]
      ["--","compute","--","--tags","foo"] -> ["compute","--tags","foo"]
    """
    out = list(argv)
    # drop one leading "--"
    if out and out[0] == "--":
        out = out[1:]
    # collapse "<cmd> -- ..." to "<cmd> ..."
    if len(out) >= 2 and out[1] == "--":
        out = [out[0]] + out[2:]
    return out


def _print_top_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help(sys.stderr)
    print(
        r"""
Examples:

  # 1) Heavy quantum step:
  #    - per-cluster δ (ppm), optional J couplings (Hz),
  #      ddCOSMO single-point energies (Hartree)
  #    - can try GPU SCF via gpu4pyscf
  ./f_predict_shifts.py compute -- \
      --tags aspirin_neutral_cdcl3 \
      --xc b3lyp \
      --basis def2-tzvp \
      --solvent CDCl3 \
      --gpu auto \
      --keep-isotopes 1H \
      --require-j

  # 2) Fast averaging step:
  #    - no new DFT
  #    - Boltzmann or MD weights → ensemble-avg δ and ⟨J⟩
  ./f_predict_shifts.py average -- \
      --tags aspirin_neutral_cdcl3 \
      --solvent CDCl3 \
      --temp 298.15
"""
    )


def main() -> None:
    prog = "f_predict_shifts.py"

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "NMR shift / scalar J prediction pipeline.\n"
            "Subcommands:\n"
            "  compute  → per-cluster DFT (σ→δ, optional J, energies)\n"
            "  average  → Boltzmann/MD fast averaging (δ, J)\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser(
        "compute",
        help=(
            "Run DFT per cluster: shieldings→δ, optional spin–spin J, "
            "and ddCOSMO single-point energies."
        ),
    )
    sub.add_parser(
        "average",
        help=(
            "Boltzmann/MD-weighted averaging of precomputed δ and J "
            "across clusters."
        ),
    )

    # no args at all → show global help + usage examples
    if len(sys.argv) == 1:
        _print_top_help(parser)
        return

    # normalize argv
    argv = _strip_separators(sys.argv[1:])
    if not argv:
        _print_top_help(parser)
        return

    cmd = argv[0]
    rest = argv[1:]

    # in case user did "<cmd> -- --tags ...", collapse again
    rest = _strip_separators([cmd] + rest)[1:]

    # map subcommand → module
    dispatch = {
        "compute": ("f_predict_shifts_compute", "f_predict_shifts_compute.py"),
        "average": ("f_predict_shifts_average", "f_predict_shifts_average.py"),
    }

    if cmd not in dispatch:
        _print_top_help(parser)
        print(
            f"\n[error] Unknown subcommand: {cmd!r}. "
            "Use 'compute' or 'average'."
        )
        return

    (module_name, child_prog) = dispatch[cmd]

    # Lazy import so just asking for help doesn't pull in PySCF/gpu4pyscf
    module = __import__(module_name, fromlist=["main"])

    # Handoff: pretend we're running that module directly so its argparse
    # sees the right argv[0].
    sys.argv = [child_prog] + rest
    return module.main()


if __name__ == "__main__":
    main()
