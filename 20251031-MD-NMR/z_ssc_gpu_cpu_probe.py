#!/usr/bin/env python3
"""
GPU SCF → CPU SSC probe (PySCF + gpu4pyscf + pyscf-properties)

- SCF: gpu4pyscf.dft.rks (B3LYP/def2-SVP)
- SSC: pyscf.prop.ssc.* on CPU
- Converts returned 3x3 reduced-coupling tensor K (a.u.) to scalar J_iso (Hz)

Expected: prints versions, CUDA accelerators, GPU energy, and a small J matrix in Hz.

Notes:
- Keep SCF on GPU for speed, but copy MO arrays to CPU before SSC.
- The numeric J table PySCF prints to stdout should agree with our computed J_iso within ~roundoff.
"""

from __future__ import annotations

import importlib
import importlib.util as iu
import sys

import numpy as np


# --- Versions / accelerators banner -----------------------------------------
def banner():
    import pyscf
    print(f"PySCF: {pyscf.__version__}")
    try:
        gpu4 = importlib.import_module("gpu4pyscf")
        print(f"gpu4pyscf: {getattr(gpu4, '__version__', 'unknown')}")
    except Exception as e:
        print(f"gpu4pyscf: import failed: {e}")

    try:
        import cupy as cp
        acc = None
        if iu.find_spec("cupy._core"):
            from cupy._core import _accelerator as accmod
            acc = accmod.get_routine_accelerators()
        ndev = cp.cuda.runtime.getDeviceCount()
        print(f"CuPy: {cp.__version__} | CUDA devices: {ndev} | accelerators: {acc}")
    except Exception as e:
        print(f"CuPy probe failed: {e}")


# --- Build molecule ----------------------------------------------------------
def build_mol():
    """Return a simple test molecule. Switch to H2O if you prefer."""
    from pyscf import gto
    # H2 (helps to see an H–H coupling)
    geo = "H 0 0 0; H 0 0 0.74"
    # alt: H2O
    # geo = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    mol = gto.M(atom=geo, basis="def2-svp").build()
    return mol


# --- SCF on GPU --------------------------------------------------------------
def run_gpu_rks(mol, xc="b3lyp", conv_tol=1e-9, max_cycle=50):
    from gpu4pyscf.dft import rks as grks
    mf = grks.RKS(mol).set(xc=xc, conv_tol=conv_tol, max_cycle=max_cycle)
    e = mf.kernel()
    return mf, float(e), bool(getattr(mf, "converged", True))


# --- Copy CuPy arrays to NumPy & prepare a CPU RKS with those orbitals -------
def mf_gpu_to_cpu_rks(mf_gpu):
    import cupy as cp
    from pyscf import dft

    def to_numpy(x):
        return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

    mf_cpu = dft.RKS(mf_gpu.mol)
    mf_cpu.xc = mf_gpu.xc
    mf_cpu.mo_coeff = to_numpy(mf_gpu.mo_coeff)
    mf_cpu.mo_occ = to_numpy(mf_gpu.mo_occ)
    mf_cpu.mo_energy = to_numpy(mf_gpu.mo_energy)
    mf_cpu.e_tot = float(getattr(mf_gpu, "e_tot", np.nan))
    # Mark as "converged" so property routines don't try to re-run SCF
    mf_cpu.converged = True
    return mf_cpu


# --- Robust import of SSC class ---------------------------------------------
def get_ssc_class(mf_cpu):
    """
    Prefer the generic SpinSpinCoupling if available; fall back to RKS/RHF-specific class.
    """
    from pyscf.prop import ssc as ssc_mod
    if hasattr(ssc_mod, "SpinSpinCoupling"):
        return ssc_mod.SpinSpinCoupling, ssc_mod
    # fallbacks for older layouts
    try:
        from pyscf.prop.ssc import rks as ssc_rks
        return ssc_rks.SSC, ssc_mod
    except Exception:
        from pyscf.prop.ssc import rhf as ssc_rhf
        return ssc_rhf.SSC, ssc_mod


# --- Convert 3x3 K tensor (a.u.) → scalar J_iso matrix (Hz) ------------------
def tensorK_to_Jiso_Hz(mol, K_au, ssc_mod):
    """
    K_au: (natm, natm, 3, 3) reduced coupling tensor (atomic units).
    Returns: J_iso_Hz: (natm, natm) scalar J matrix in Hz.
    """
    # 1) Isotropic reduced coupling
    Kiso_au = np.trace(K_au, axis1=2, axis2=3) / 3.0  # (natm, natm)

    # 2) Nuclear g-factors
    g = None
    if hasattr(ssc_mod, "nuc_g_factor"):
        gf = ssc_mod.nuc_g_factor
        try:
            # index by atomic number if it's an array-like
            g = np.array([gf[Z] for Z in mol.atom_charges()], float)
        except Exception:
            # try dict by element symbol
            g = np.array([gf[mol.atom_symbol(i)] for i in range(mol.natm)], float)
    if g is None:
        # minimal fallback (covers typical NMR-active nuclei; extend as needed)
        fallback = {
            "H": 5.5856946893, "F": 5.257731, "P": 2.2632, "C": 1.404825, "N": -0.566378,
            "Si": -1.1106, "O": -1.89379,
        }
        g = np.array([fallback[mol.atom_symbol(i)] for i in range(mol.natm)], float)

    # 3) a.u. → Hz conversion for reduced coupling K
    if hasattr(ssc_mod, "K_au2Hz"):
        au2Hz = float(ssc_mod.K_au2Hz)
    elif hasattr(ssc_mod, "K_au2MHz"):
        au2Hz = float(ssc_mod.K_au2MHz) * 1e6
    else:
        raise RuntimeError("Cannot find K_au→Hz conversion constant in pyscf.prop.ssc")

    # 4) Scalar J
    Kiso_Hz = Kiso_au * au2Hz
    Jiso_Hz = (g[:, None] * g[None, :]) * Kiso_Hz
    return Jiso_Hz


# --- Main --------------------------------------------------------------------
def main():
    banner()
    mol = build_mol()

    # GPU SCF
    mf_gpu, e_gpu, conv = run_gpu_rks(mol)
    print(f"\nGPU RKS energy: {e_gpu:.10f}  | converged={conv}")

    # Ensure CPU copy of MOs for properties
    mf_cpu = mf_gpu_to_cpu_rks(mf_gpu)

    # SSC kernel (CPU)
    SSC, ssc_mod = get_ssc_class(mf_cpu)
    ssc = SSC(mf_cpu)
    K_au = ssc.kernel()  # shape: (natm, natm, 3, 3)

    # Convert to scalar J in Hz
    J_Hz = tensorK_to_Jiso_Hz(mol, np.asarray(K_au), ssc_mod)

    # Pretty print small J matrix
    with np.printoptions(precision=5, suppress=True):
        print("\nJ_iso (Hz):")
        print(J_Hz)

    # For convenience, list just the H–H entries if present
    idx_H = [i for i in range(mol.natm) if mol.atom_symbol(i) == "H"]
    if len(idx_H) >= 2:
        sub = J_Hz[np.ix_(idx_H, idx_H)]
        print("\nJ_iso (Hz) — H/H submatrix:")
        print(sub)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

