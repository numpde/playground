#!/usr/bin/env python3
"""
z_diag_ssc_stack.py
Purpose-built diagnostics for the PySCF ⇄ gpu4pyscf ⇄ CuPy ⇄ SSC stack.

What it checks (with PASS/FAIL/SKIP):
  1) Environment banner: versions, CUDA device count, accelerators
  2) CPU baseline: SCF(B3LYP/def2-SVP) on H2 → SSC → parse J(Hz)
  3) GPU path: SCF on GPU (if available) → copy MOs to CPU → SSC → parse J(Hz)
  4) SSC return-shape handling: (natm,natm,3,3) | (npair,3,3) | (3,3)
  5) If K→Hz constants present: derive J_iso(Hz) from K and compare to parsed table
  6) Optional large-molecule smoke test (short SCF) to surface segfaults

Usage examples:
  python z_diag_ssc_stack.py
  python z_diag_ssc_stack.py --gpu auto
  python z_diag_ssc_stack.py --probe-xyz strychnine_cluster0.xyz --cycles 6
  python z_diag_ssc_stack.py --probe-pdb strychnine_neutral_cdcl3_cluster_0_rep.pdb --cycles 6
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util as iu
import io
import os
import sys
import textwrap
import warnings
from contextlib import redirect_stdout

import numpy as np
from pyscf import gto, dft

# Silence PySCF property-module “under testing / not fully tested” notices
warnings.filterwarnings(
    "ignore",
    message=r"Module .* (is under testing|is not fully tested)",
    category=UserWarning,
    module=r"^pyscf\.prop\..*",
)


# ---------- small util ----------

def status(tag: str, ok: bool | None, note: str = ""):
    flag = "PASS" if ok is True else ("FAIL" if ok is False else "SKIP")
    msg = f"[{flag:<4}] {tag}"
    if note:
        msg += f" — {note}"
    print(msg)


def have(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def getenv(name: str, default="(unset)"):
    v = os.environ.get(name, default)
    return v if v != "" else "(empty)"


# ---------- banner ----------

def banner():
    import pyscf
    print("=== Environment ===")
    print(f"python: {sys.version.split()[0]} | exe: {sys.executable}")
    print(f"PySCF: {pyscf.__version__} | file: {pyscf.__file__}")
    print(f"CUPY_ACCELERATORS: {getenv('CUPY_ACCELERATORS')}")
    print(f"CUDA_VISIBLE_DEVICES: {getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"PYSCF_MAX_MEMORY: {getenv('PYSCF_MAX_MEMORY')}")
    # CuPy / accelerators
    try:
        import cupy as cp
        acc = None
        if iu.find_spec("cupy._core"):
            from cupy._core import _accelerator as accmod
            acc = accmod.get_routine_accelerators()
        ndev = cp.cuda.runtime.getDeviceCount()
        print(f"CuPy: {cp.__version__} | devices: {ndev} | accelerators: {acc}")
    except Exception as e:
        print(f"CuPy probe: {e}")
    # gpu4pyscf
    try:
        gpu4 = importlib.import_module("gpu4pyscf")
        print(f"gpu4pyscf: {getattr(gpu4, '__version__', 'unknown')} | file: {gpu4.__file__}")
    except Exception as e:
        print(f"gpu4pyscf import: {e}")


# ---------- geometry loaders ----------

def mol_h2():
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp").build()


def mol_from_xyz(path: str, basis: str):
    with open(path, "r") as f:
        raw = f.read().strip().splitlines()
    # accept simple XYZ (skip first 2 header lines if present)
    if len(raw) >= 2 and raw[0].strip().isdigit():
        raw = raw[2:]
    atom_lines = []
    for line in raw:
        if not line.strip():
            continue
        toks = line.split()
        if len(toks) < 4:  # skip invalid
            continue
        sym, x, y, z = toks[:4]
        atom_lines.append(f"{sym} {x} {y} {z}")
    geo = "; ".join(atom_lines)
    return gto.M(atom=geo, basis=basis).build()


def mol_from_pdb(path: str, basis: str):
    # Prefer RDKit if present; otherwise try a minimal PDB ATOM parser.
    if have("rdkit"):
        from rdkit import Chem
        m = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        if m is None:
            raise ValueError("RDKit failed to read PDB")
        conf = m.GetConformer()
        lines = []
        for i, a in enumerate(m.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            lines.append(f"{a.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}")
        geo = "; ".join(lines)
        return gto.M(atom=geo, basis=basis).build()
    # Fallback: parse ATOM/HETATM records with element symbol in column 77–78
    atoms = []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                sym = (line[76:78].strip() or line[12:16].strip()[0]).strip()
                x = float(line[30:38]);
                y = float(line[38:46]);
                z = float(line[46:54])
                atoms.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
            except Exception:
                continue
    if not atoms:
        raise ValueError("No atoms parsed from PDB")
    geo = "; ".join(atoms)
    return gto.M(atom=geo, basis=basis).build()


# ---------- GPU SCF and CPU hand-off ----------

def to_numpy_maybe(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return x


def run_scf(mol, xc: str, conv_tol: float, max_cycle: int, gpu_mode: str):
    """gpu_mode: 'on' | 'off' | 'auto'"""
    mf = None
    used_gpu = False
    if gpu_mode in ("on", "auto") and have("gpu4pyscf.dft.rks"):
        try:
            from gpu4pyscf.dft import rks as grks
            mf = grks.RKS(mol).set(xc=xc, conv_tol=conv_tol, max_cycle=max_cycle)
            e = float(mf.kernel())
            used_gpu = True
            return mf, e, used_gpu, True
        except Exception as e:
            status("GPU SCF", False, f"{type(e).__name__}: {e}")
            if gpu_mode == "on":
                return None, np.nan, True, False
            # fall through to CPU
    # CPU
    try:
        mf = dft.RKS(mol).set(xc=xc, conv_tol=conv_tol, max_cycle=max_cycle)
        e = float(mf.kernel())
        return mf, e, False, True
    except Exception as e:
        return None, np.nan, False, False


def handoff_to_cpu(mf_any):
    """Build a CPU RKS with MO arrays copied (avoid re-running SCF for SSC)."""
    mfc = dft.RKS(mf_any.mol)
    mfc.xc = mf_any.xc
    mfc.mo_coeff = to_numpy_maybe(mf_any.mo_coeff)
    mfc.mo_occ = to_numpy_maybe(mf_any.mo_occ)
    mfc.mo_energy = to_numpy_maybe(mf_any.mo_energy)
    mfc.e_tot = float(getattr(mf_any, "e_tot", np.nan))
    mfc.converged = True
    return mfc


# ---------- SSC + parsing + optional K→J ----------

def get_ssc_class():
    from pyscf.prop import ssc as ssc_mod
    if hasattr(ssc_mod, "SpinSpinCoupling"):
        return ssc_mod.SpinSpinCoupling, ssc_mod
    # fallbacks
    try:
        from pyscf.prop.ssc import rks as ssc_rks
        return ssc_rks.SSC, ssc_mod
    except Exception:
        from pyscf.prop.ssc import rhf as ssc_rhf
        return ssc_rhf.SSC, ssc_mod


def parse_j_table(stdout_text: str):
    """Parse the 'Spin-spin coupling constant J (Hz)' table from captured stdout."""
    anchor = "Spin-spin coupling constant J (Hz)"
    i = stdout_text.find(anchor)
    if i == -1:
        return [], {}
    lines = stdout_text[i:].splitlines()[1:]  # skip title
    block = []
    for l in lines:
        if not l.strip():
            break
        block.append(l.rstrip("\n"))
    if not block:
        return [], {}
    # header with '#0  #1  ...'
    header = next((l for l in block if "#" in l), "")
    import re
    cols = [int(x) for x in re.findall(r"#(\d+)", header)]
    data = {}
    start = (block.index(header) + 1) if header and header in block else 0
    for l in block[start:]:
        m = re.match(r"^\s*(\d+)\s+\S+\s+(.*)$", l)
        if not m:  # tolerate blank/stray lines
            continue
        row = int(m.group(1))
        floats = [float(x) for x in m.group(2).split()]
        if not cols:
            cols = list(range(len(floats)))
        for j, val in enumerate(floats):
            if j < len(cols):
                data[(row, cols[j])] = val
    return cols, data


def assemble_square_from_entries(n: int, entries: dict[tuple[int, int], float]) -> np.ndarray:
    M = np.zeros((n, n), float)
    for (i, j), v in entries.items():
        M[i, j] = v
        M[j, i] = v
    return M


def nuc_g_vector(mol, ssc_mod):
    # Use table if present, else minimal fallback
    if hasattr(ssc_mod, "nuc_g_factor"):
        gf = ssc_mod.nuc_g_factor
        try:
            return np.array([gf[Z] for Z in mol.atom_charges()], float)
        except Exception:
            return np.array([gf[mol.atom_symbol(i)] for i in range(mol.natm)], float)
    fallback = {"H": 5.5856946893, "C": 1.404825, "N": -0.566378, "O": -1.89379, "F": 5.257731}
    return np.array([fallback.get(mol.atom_symbol(i), 0.0) for i in range(mol.natm)], float)


def K_au_to_Jiso_Hz(mol, ssc_obj, K_au, ssc_mod):
    """Best-effort conversion if constants exist; else raises."""
    # constants (prefer Hz if available)
    if hasattr(ssc_mod, "K_au2Hz"):
        au2Hz = float(ssc_mod.K_au2Hz)
    elif hasattr(ssc_mod, "K_au2MHz"):
        au2Hz = float(ssc_mod.K_au2MHz) * 1e6
    else:
        raise RuntimeError("No K→Hz/MHz constant in pyscf.prop.ssc")
    # pairs
    with_ssc = getattr(ssc_obj, "with_ssc", ssc_obj)
    pairs = getattr(with_ssc, "nuc_pair", None)
    if not pairs:
        natm = mol.natm
        pairs = [(i, j) for i in range(natm) for j in range(i + 1, natm)]
    # normalize shapes
    K = np.asarray(K_au)
    natm = mol.natm
    triplets = []
    if K.ndim == 4 and K.shape[2:] == (3, 3):
        for i in range(natm):
            for j in range(natm):
                triplets.append((i, j, K[i, j]))
    elif K.ndim == 3 and K.shape[-2:] == (3, 3):
        if len(pairs) != K.shape[0]:
            raise ValueError(f"K/pairs mismatch: {K.shape[0]} vs {len(pairs)}")
        for (i, j), Tij in zip(pairs, K):
            triplets.append((i, j, Tij));
            triplets.append((j, i, Tij.T))
    elif K.ndim == 2 and K.shape == (3, 3):
        (i, j) = pairs[0] if len(pairs) else (0, 1)
        triplets.append((i, j, K));
        triplets.append((j, i, K.T))
    else:
        raise ValueError(f"Unexpected K shape: {K.shape}")
    # accumulate K_iso and convert
    Kiso_Hz = np.zeros((natm, natm), float)
    for i, j, Tij in triplets:
        Kiso_Hz[i, j] += float(np.trace(Tij) / 3.0) * au2Hz
    g = nuc_g_vector(mol, ssc_mod)
    Jiso_Hz = (g[:, None] * g[None, :]) * Kiso_Hz
    return Jiso_Hz


def run_ssc_and_parse(mf_cpu):
    SSC, ssc_mod = get_ssc_class()
    ssc = SSC(mf_cpu)
    buf = io.StringIO()
    with redirect_stdout(buf):
        K = ssc.kernel()
    out = buf.getvalue()
    cols, entries = parse_j_table(out)
    J_from_table = None
    if entries:
        n = 1 + max(max(i, j) for (i, j) in entries.keys())
        J_from_table = assemble_square_from_entries(n, entries)
    return ssc, ssc_mod, K, J_from_table, out


# ---------- main flow ----------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent(__doc__))
    ap.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", default="b3lyp")
    ap.add_argument("--cycles", type=int, default=50, help="SCF max_cycle for small test")
    ap.add_argument("--probe-xyz", default=None, help="Optional large-molecule XYZ for smoke test")
    ap.add_argument("--probe-pdb", default=None, help="Optional large-molecule PDB for smoke test")
    ap.add_argument("--probe-cycles", type=int, default=6, help="Short cycles for smoke test")
    args = ap.parse_args()

    banner()

    # 1) CPU baseline on H2
    print("\n=== Baseline: CPU SCF → SSC (H2) ===")
    mol = mol_h2()
    mf, e, used_gpu, ok = run_scf(mol, args.xc, 1e-10, args.cycles, gpu_mode="off")
    status("CPU SCF(H2)", ok, f"energy={e:.10f}" if ok else "")
    if not ok:
        return sys.exit(2)
    mf_cpu = handoff_to_cpu(mf)
    try:
        ssc, ssc_mod, K, J_tab, out = run_ssc_and_parse(mf_cpu)
        status("SSC kernel(H2)", True, f"K shape={np.shape(K)}; table={'yes' if J_tab is not None else 'no'}")
        if J_tab is not None:
            print("J_table(Hz) small matrix:\n", np.array_str(J_tab, precision=6, suppress_small=True))
        # optional K→J
        try:
            J_conv = K_au_to_Jiso_Hz(mol, ssc, K, ssc_mod)
            status("K→J conv(H2)", True)
            if J_tab is not None and J_conv.shape == J_tab.shape:
                diff = np.nanmax(np.abs(J_conv - J_tab))
                status("Compare conv vs table(H2)", True, f"max|Δ|={diff:.3e} Hz")
        except Exception as e:
            status("K→J conv(H2)", None, f"{type(e).__name__}: {e}")
    except Exception as e:
        status("SSC kernel(H2)", False, f"{type(e).__name__}: {e}")

    # 2) GPU path on H2 (if available)
    print("\n=== GPU path: GPU SCF → CPU SSC (H2) ===")
    mf, e, used_gpu, ok = run_scf(mol, args.xc, 1e-9, args.cycles, gpu_mode=args.gpu)
    status("GPU SCF(H2)" if used_gpu else "GPU SCF(H2)", ok,
           ("energy={:.10f}".format(e) if ok else (
               "gpu4pyscf unavailable" if not have("gpu4pyscf.dft.rks") else "error")))
    if ok:
        try:
            mf_cpu = handoff_to_cpu(mf)
            ssc, ssc_mod, K, J_tab, out = run_ssc_and_parse(mf_cpu)
            status("SSC kernel(H2, from GPU MOs)", True,
                   f"K shape={np.shape(K)}; table={'yes' if J_tab is not None else 'no'}")
            if J_tab is not None:
                print("J_table(Hz) small matrix (GPU MOs):\n", np.array_str(J_tab, precision=6, suppress_small=True))
        except Exception as e:
            status("SSC kernel(H2, from GPU MOs)", False, f"{type(e).__name__}: {e}")
    else:
        status("GPU path skipped", None, "No GPU or gpu4pyscf error")

    # 3) Optional large-molecule smoke test
    probe_mol = None
    src = None
    try:
        if args.probe_xyz:
            probe_mol = mol_from_xyz(args.probe_xyz, args.basis);
            src = f"XYZ:{args.probe_xyz}"
        elif args.probe_pdb:
            probe_mol = mol_from_pdb(args.probe_pdb, args.basis);
            src = f"PDB:{args.probe_pdb}"
    except Exception as e:
        status("Load probe molecule", False, f"{type(e).__name__}: {e}")

    if probe_mol is not None:
        print(f"\n=== Smoke test: {src} | natm={probe_mol.natm} ===")
        # CPU short SCF
        mf, e, used_gpu, ok = run_scf(probe_mol, args.xc, 1e-6, args.probe_cycles, gpu_mode="off")
        status("CPU SCF(probe, short)", ok, f"E={e:.6f}" if ok else "")
        # GPU short SCF (to reproduce crashes quickly)
        mf, e, used_gpu, ok = run_scf(probe_mol, args.xc, 1e-6, args.probe_cycles, gpu_mode=args.gpu)
        status("GPU SCF(probe, short)" if used_gpu else "GPU SCF(probe, short)", ok,
               f"E={e:.6f}" if ok else "error (see above)")
        if ok:
            try:
                mf_cpu = handoff_to_cpu(mf)
                ssc, ssc_mod, K, J_tab, out = run_ssc_and_parse(mf_cpu)
                status("SSC kernel(probe, from GPU MOs)", True,
                       f"K shape={np.shape(K)}; table={'yes' if J_tab is not None else 'no'}")
                if J_tab is not None:
                    print("J_table(Hz) probe (head):")
                    # print only a small top-left block to keep output readable
                    m = min(8, J_tab.shape[0])
                    print(np.array_str(J_tab[:m, :m], precision=3, suppress_small=True))
            except Exception as e:
                status("SSC kernel(probe, from GPU MOs)", False, f"{type(e).__name__}: {e}")
    else:
        status("Smoke test", None, "no --probe-xyz/--probe-pdb provided")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
