#!/usr/bin/env python3
"""
z_ssc_gpu_cpu_probe.py

Probe SSC on CPU and GPU→CPU paths and *extract J(Hz)* by parsing PySCF's
own printed table. This avoids relying on internal, non-public conversion
constants from K (a.u.) → J (Hz).

Usage:
  python z_ssc_gpu_cpu_probe.py [--probe-xyz path | --probe-pdb path]
"""

from __future__ import annotations
import io
import os
import re
import sys
import argparse
import warnings
from contextlib import redirect_stdout

# Silence only the noisy "under testing / not fully tested" notices from pyscf.prop.*
warnings.filterwarnings(
    "ignore",
    message=r"Module .* (is under testing|is not fully tested)",
    category=UserWarning,
    module=r"^pyscf\.prop\..*",
)

def _env_summary():
    def modver(name):
        try:
            m = __import__(name)
            return getattr(m, "__version__", "unknown"), getattr(m, "__file__", None)
        except Exception:
            return None, None

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    exe = sys.executable
    pyscf_v, pyscf_f = modver("pyscf")
    cupy_v, _ = modver("cupy")
    g4_v, g4_f = modver("gpu4pyscf")

    # CuPy accelerators / device probe (best-effort)
    accelerators = None
    ndev = None
    try:
        import cupy as cp
        from cupy._core import _accelerator as acc
        accelerators = list(acc.get_routine_accelerators())
        try:
            ndev = cp.cuda.runtime.getDeviceCount()
        except Exception:
            ndev = "unavailable"
    except Exception:
        pass

    print("=== Environment ===")
    print(f"python: {py} | exe: {exe}")
    print(f"PySCF: {pyscf_v} | file: {pyscf_f}")
    print(f"CUPY_ACCELERATORS: {os.environ.get('CUPY_ACCELERATORS','(unset)')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}")
    print(f"PYSCF_MAX_MEMORY: {os.environ.get('PYSCF_MAX_MEMORY','(unset)')}")
    print(f"CuPy: {cupy_v} | devices: {ndev} | accelerators: {accelerators}")
    print(f"gpu4pyscf: {g4_v} | file: {g4_f}")
    print()

def _parse_j_table(stdout_text: str) -> tuple[list[str], list[list[float]]]:
    """
    Parse the 'Spin-spin coupling constant J (Hz)' table from PySCF's SSC stdout.

    Returns: (labels, J_matrix)
      labels: e.g. ["H0","H1", ...] using the element + row index
      J_matrix: NxN floats (Hz)

    Raises ValueError if the table can't be found or parsed.
    """
    # Find the J-table block
    anchor = re.search(r"Spin-spin coupling constant J \(Hz\)\s*\n", stdout_text)
    if not anchor:
        raise ValueError("J(Hz) table block not found in SSC stdout.")

    # The header line with '#0  #1  ...'
    header_match = re.search(r"^\s*(#\d+(?:\s+#\d+)*)\s*$", stdout_text[anchor.end():], re.MULTILINE)
    if not header_match:
        raise ValueError("Header with column indices not found in J(Hz) block.")
    header_cols = re.findall(r"#(\d+)", header_match.group(1))
    n = len(header_cols)
    if n == 0:
        raise ValueError("No column indices in J(Hz) header.")

    # Collect subsequent N lines that start with row index + element
    # Example line:
    # "        1 H  324.88995   0.00000"
    # We accept scientific notation as well.
    block_start = anchor.end() + header_match.end()
    lines = stdout_text[block_start:].splitlines()

    num_pat = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    row_pat = re.compile(rf"^\s*(\d+)\s+([A-Za-z]+)\s+((?:{num_pat}\s+)+)")

    rows = []
    for line in lines:
        m = row_pat.match(line)
        if not m:
            # stop once rows stop
            if rows:
                break
            else:
                continue
        idx = int(m.group(1))
        elem = m.group(2)
        nums = [float(x) for x in re.findall(num_pat, m.group(3))]
        if len(nums) != n:
            raise ValueError(f"Row {idx} has {len(nums)} values; expected {n}.")
        rows.append((idx, elem, nums))
        if len(rows) == n:
            break

    if len(rows) != n:
        raise ValueError(f"Parsed {len(rows)} rows; expected {n}.")

    # Order rows by their (parsed) index and build labels + matrix
    rows.sort(key=lambda t: t[0])
    labels = [f"{elem}{idx}" for (idx, elem, _) in rows]
    J = [nums for (_, _, nums) in rows]
    # Ensure symmetry by mirroring max(|J_ij|, |J_ji|) with sign of J_ij
    for i in range(n):
        for j in range(n):
            if j < i:
                a, b = J[i][j], J[j][i]
                J[i][j] = a if abs(a) >= abs(b) else (b if a >= 0 else -b)
    return labels, J

def _cpu_scf_h2_and_ssc():
    from pyscf import gto, dft
    from pyscf.prop.ssc import rhf as ssc_rhf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp").build()
    mf = dft.RKS(mol).set(xc="b3lyp", conv_tol=1e-10, max_cycle=50)

    e = mf.kernel()
    if not getattr(mf, "converged", True):
        raise RuntimeError("CPU SCF(H2) did not converge")

    print(f"[PASS] CPU SCF(H2) — energy={e:.10f}")

    ssc = ssc_rhf.SSC(mf)
    s = io.StringIO()
    with redirect_stdout(s):
        K = ssc.kernel()  # prints both K and J tables
    out = s.getvalue()

    if not (hasattr(K, "shape") and K.shape[-2:] == (3, 3)):
        raise RuntimeError("SSC kernel(H2) did not return a (..,3,3) tensor K")
    print(f"[PASS] SSC kernel(H2) — K shape={getattr(K,'shape',None)}; table=printed")

    try:
        labels, J = _parse_j_table(out)
        maxabs = max(abs(x) for row in J for x in row)
        print(f"[PASS] Parsed J(Hz) from stdout — labels={labels}; max|J|={maxabs:.6f} Hz")
    except Exception as ex:
        print(f"[WARN] Could not parse J(Hz) table: {ex}")
        print("[INFO] Emitting SSC stdout for inspection:\n")
        print(out)

def _gpu_scf_h2_then_cpu_ssc():
    from pyscf import gto, dft
    from gpu4pyscf.dft import rks as grks
    from pyscf.prop.ssc import rhf as ssc_rhf
    import numpy as np
    import cupy

    def to_numpy(x):
        return x.get() if isinstance(x, cupy.ndarray) else x

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp").build()

    # GPU SCF
    mf_gpu = grks.RKS(mol).set(xc="b3lyp", conv_tol=1e-9, max_cycle=50)
    e = mf_gpu.kernel()
    if not getattr(mf_gpu, "converged", True):
        print("[WARN] GPU SCF not converged — restarting on CPU.")
        mf_gpu = dft.RKS(mol).set(xc="b3lyp", conv_tol=1e-10, max_cycle=50).run()
    print(f"[PASS] GPU SCF(H2) — energy={mf_gpu.e_tot:.10f}")

    # Map CuPy → NumPy and run SSC on CPU
    mf_cpu = dft.RKS(mol)
    mf_cpu.xc = mf_gpu.xc
    mf_cpu.mo_coeff  = to_numpy(mf_gpu.mo_coeff)
    mf_cpu.mo_occ    = to_numpy(mf_gpu.mo_occ)
    mf_cpu.mo_energy = to_numpy(mf_gpu.mo_energy)
    mf_cpu.e_tot     = mf_gpu.e_tot

    ssc = ssc_rhf.SSC(mf_cpu)
    s = io.StringIO()
    with redirect_stdout(s):
        K = ssc.kernel()
    out = s.getvalue()

    if not (hasattr(K, "shape") and K.shape[-2:] == (3, 3)):
        raise RuntimeError("SSC kernel(H2, from GPU MOs) did not return a (..,3,3)")
    print(f"[PASS] SSC kernel(H2, from GPU MOs) — K shape={getattr(K,'shape',None)}; table=printed")

    try:
        labels, J = _parse_j_table(out)
        maxabs = max(abs(x) for row in J for x in row)
        print(f"[PASS] Parsed J(Hz) from stdout (GPU→CPU) — labels={labels}; max|J|={maxabs:.6f} Hz")
    except Exception as ex:
        print(f"[WARN] Could not parse J(Hz) table (GPU→CPU): {ex}")
        print("[INFO] Emitting SSC stdout for inspection:\n")
        print(out)

def _smoke_test_from_file(path: str):
    """
    Minimal smoke test: load a structure and see if CPU SCF runs,
    then attempt SSC and parse J(Hz) if available.
    """
    from pyscf import gto, dft
    from pyscf.prop.ssc import rhf as ssc_rhf

    ext = os.path.splitext(path)[1].lower()
    if ext == ".xyz":
        mol = gto.Mole(atom=open(path).read(), unit="Angstrom", basis="def2-svp")
    else:
        # relax: PDB or others routed via .fromfile when possible
        mol = gto.Mole()
        mol.build(atom=path, basis="def2-svp")  # PySCF can read simple PDB via filename

    mf = dft.RKS(mol).set(xc="b3lyp", conv_tol=1e-8, max_cycle=100)
    e = mf.kernel()
    if not getattr(mf, "converged", True):
        print(f"[FAIL] CPU SCF({os.path.basename(path)}) did not converge")
        return
    print(f"[PASS] CPU SCF({os.path.basename(path)}) — energy={e:.8f}")

    ssc = ssc_rhf.SSC(mf)
    s = io.StringIO()
    with redirect_stdout(s):
        try:
            ssc.kernel()
        except Exception as ex:
            print(f"[WARN] SSC kernel failed on {os.path.basename(path)}: {ex}")
            return
    out = s.getvalue()
    try:
        labels, J = _parse_j_table(out)
        maxabs = max(abs(x) for row in J for x in row)
        print(f"[PASS] Parsed J(Hz) — N={len(labels)}; max|J|={maxabs:.6f} Hz")
    except Exception as ex:
        print(f"[WARN] Could not parse J(Hz) table: {ex}")
        print("[INFO] Emitting SSC stdout for inspection:\n")
        print(out)

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--probe-xyz", type=str, help="Path to .xyz")
    g.add_argument("--probe-pdb", type=str, help="Path to .pdb")
    args = p.parse_args()

    _env_summary()

    print("=== Baseline: CPU SCF → SSC (H2) ===")
    _cpu_scf_h2_and_ssc()
    print()

    print("=== GPU path: GPU SCF → CPU SSC (H2) ===")
    _gpu_scf_h2_then_cpu_ssc()
    print()

    if args.probe_xyz or args.probe_pdb:
        path = args.probe_xyz or args.probe_pdb
        print(f"=== Smoke test on user file: {path} ===")
        _smoke_test_from_file(path)
    else:
        print("[SKIP] Smoke test — no --probe-xyz/--probe-pdb provided")
    print("\nDone.")

if __name__ == "__main__":
    main()
