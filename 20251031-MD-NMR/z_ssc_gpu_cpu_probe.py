#!/usr/bin/env python3
"""
GPU SCF → CPU SSC probe (robust shapes)

- SCF (GPU): gpu4pyscf.dft.rks.RKS (B3LYP/def2-SVP)
- SSC (CPU): pyscf.prop.ssc.* ; converts returned reduced-coupling tensor K (a.u.)
             to scalar J_iso matrix (Hz), handling (natm,natm,3,3), (npair,3,3), or (3,3).

Run inside your md-nmr env.
"""

from __future__ import annotations

import importlib
import importlib.util as iu
import sys

import numpy as np


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


def build_mol():
    from pyscf import gto
    geo = "H 0 0 0; H 0 0 0.74"  # simple H–H J
    mol = gto.M(atom=geo, basis="def2-svp").build()
    return mol


def run_gpu_rks(mol, xc="b3lyp", conv_tol=1e-9, max_cycle=50):
    from gpu4pyscf.dft import rks as grks
    mf = grks.RKS(mol).set(xc=xc, conv_tol=conv_tol, max_cycle=max_cycle)
    e = mf.kernel()
    return mf, float(e), bool(getattr(mf, "converged", True))


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
    mf_cpu.converged = True
    return mf_cpu


def get_ssc_class():
    # Prefer generic; fall back to RKS/RHF
    from pyscf.prop import ssc as ssc_mod
    if hasattr(ssc_mod, "SpinSpinCoupling"):
        return ssc_mod.SpinSpinCoupling, ssc_mod
    try:
        from pyscf.prop.ssc import rks as ssc_rks
        return ssc_rks.SSC, ssc_mod
    except Exception:
        from pyscf.prop.ssc import rhf as ssc_rhf
        return ssc_rhf.SSC, ssc_mod


def _nuc_g_vector(mol, ssc_mod):
    # Try official table; fall back to a small map for common nuclei.
    if hasattr(ssc_mod, "nuc_g_factor"):
        gf = ssc_mod.nuc_g_factor
        try:
            return np.array([gf[Z] for Z in mol.atom_charges()], float)
        except Exception:
            return np.array([gf[mol.atom_symbol(i)] for i in range(mol.natm)], float)
    fallback = {
        "H": 5.5856946893, "F": 5.257731, "P": 2.2632, "C": 1.404825, "N": -0.566378,
        "Si": -1.1106, "O": -1.89379,
    }
    return np.array([fallback[mol.atom_symbol(i)] for i in range(mol.natm)], float)


def _au2Hz_const(ssc_mod):
    if hasattr(ssc_mod, "K_au2Hz"):
        return float(ssc_mod.K_au2Hz)
    if hasattr(ssc_mod, "K_au2MHz"):
        return float(ssc_mod.K_au2MHz) * 1e6
    raise RuntimeError("Cannot find K_au→Hz conversion constant in pyscf.prop.ssc")


def assemble_Jiso_from_K(mol, ssc_obj, K_au, ssc_mod):
    """
    Handle shapes:
      (natm,natm,3,3)  – full tensor grid
      (npair,3,3)      – one tensor per requested pair
      (3,3)            – single pair tensor
    """
    natm = mol.natm
    g = _nuc_g_vector(mol, ssc_mod)
    au2Hz = _au2Hz_const(ssc_mod)

    # Make pairs list
    with_ssc = getattr(ssc_obj, "with_ssc", ssc_obj)
    pairs = getattr(with_ssc, "nuc_pair", None)
    if pairs is None or len(pairs) == 0:
        pairs = [(i, j) for i in range(natm) for j in range(i + 1, natm)]

    # Normalize K shape to a list of (i,j,Kij_3x3)
    triplets = []
    K = np.asarray(K_au)

    if K.ndim == 4 and K.shape[2:] == (3, 3):
        # Full (natm,natm,3,3)
        for i in range(natm):
            for j in range(natm):
                triplets.append((i, j, K[i, j]))
    elif K.ndim == 3 and K.shape[-2:] == (3, 3):
        # (npair,3,3): align with pairs order
        if len(pairs) != K.shape[0]:
            raise ValueError(f"Shape mismatch: {K.shape[0]} tensors but {len(pairs)} pairs")
        for (i, j), Tij in zip(pairs, K):
            triplets.append((i, j, Tij))
            triplets.append((j, i, Tij.T))  # ensure symmetry
    elif K.ndim == 2 and K.shape == (3, 3):
        # Single tensor; require exactly one pair
        if len(pairs) != 1:
            # fallback: assume first unique pair
            i, j = (0, 1) if natm >= 2 else (0, 0)
        else:
            (i, j) = pairs[0]
        triplets.append((i, j, K))
        triplets.append((j, i, K.T))
    else:
        raise ValueError(f"Unexpected K shape: {K.shape}")

    # Accumulate isotropic reduced coupling and convert to J (Hz)
    Kiso_Hz = np.zeros((natm, natm), float)
    for i, j, Tij in triplets:
        Kiso_au = np.trace(Tij) / 3.0
        Kiso_Hz[i, j] += float(Kiso_au) * au2Hz

    Jiso_Hz = (g[:, None] * g[None, :]) * Kiso_Hz
    return Jiso_Hz


def main():
    banner()
    mol = build_mol()

    mf_gpu, e_gpu, conv = run_gpu_rks(mol)
    print(f"\nGPU RKS energy: {e_gpu:.10f}  | converged={conv}")

    mf_cpu = mf_gpu_to_cpu_rks(mf_gpu)

    SSC, ssc_mod = get_ssc_class()
    ssc = SSC(mf_cpu)

    # Run kernel (will also print PySCF's J/K tables to stdout)
    K_au = ssc.kernel()

    J_Hz = assemble_Jiso_from_K(mol, ssc, K_au, ssc_mod)

    with np.printoptions(precision=5, suppress=True):
        print("\nJ_iso (Hz):")
        print(J_Hz)

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
