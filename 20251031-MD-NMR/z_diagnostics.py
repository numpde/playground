#!/usr/bin/env python3
# gpu_cutensor_diag.py — CuPy/cuTENSOR/gpu4pyscf SCF diagnostics (segfault-safe)
# - Prints CUDA/CuPy/gpu4pyscf state (versions, active accelerators, module paths)
# - Estimates AO size from a PDB (+basis) to guide GPU safety
# - Probes GPU SCF in a subprocess; detects to_gpu/kernel failures and segfaults
# - Optional stress mode (replicate CH4) and escalation sweep
# RA style: concise, clear, no fluff.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from typing import Dict, List, Optional, Tuple


# ---------- helpers ----------

def hdr(t: str) -> None:
    print(f"\n=== {t} ===")


def _get_module_path(modname: str) -> Optional[str]:
    try:
        mod = __import__(modname)
        return getattr(mod, '__file__', None)
    except Exception:
        return None


def snapshot_runtime() -> Dict[str, object]:
    info = {
        'python': sys.version.split()[0],
        'exe': sys.executable,
        'env': {k: os.environ.get(k) for k in [
            'CUPY_ACCELERATORS', 'CUDA_VISIBLE_DEVICES',
            'LD_LIBRARY_PATH', 'CONDA_PREFIX', 'PYSCF_MAX_MEMORY'
        ]},
        'pyscf_ver': None, 'pyscf_path': None,
        'gpu4pyscf_ver': None, 'gpu4pyscf_path': None,
        'cupy_ver': None, 'cupy_path': None,
        'cupy_accel_active': 'unknown',
        'cupy_show_config': None,
        'cuda_runtime_ver': None, 'cuda_driver_ver': None,
        'device_name': None, 'device_cc': None, 'mem_info': None,
        'cutensor_available': None, 'cutensor_pkg_ver': None,
        'gpu4pyscf_engine': None,
    }
    # PySCF/gpu4pyscf
    try:
        import pyscf
        info['pyscf_ver'] = getattr(pyscf, '__version__', 'unknown')
        info['pyscf_path'] = _get_module_path('pyscf')
    except Exception:
        pass
    try:
        import gpu4pyscf
        info['gpu4pyscf_ver'] = getattr(gpu4pyscf, '__version__', 'unknown')
        info['gpu4pyscf_path'] = _get_module_path('gpu4pyscf')
        try:
            import gpu4pyscf.lib.cutensor as g4ct  # type: ignore
            info['gpu4pyscf_engine'] = getattr(g4ct, 'contract_engine', None)
        except Exception:
            pass
    except Exception:
        pass
    # CuPy / CUDA
    try:
        import cupy as cp
        info['cupy_ver'] = getattr(cp, '__version__', 'unknown')
        info['cupy_path'] = _get_module_path('cupy')
        # accelerators
        accel = 'unknown'
        try:
            import cupyx  # type: ignore
            get_act = getattr(cupyx._accelerators, 'get_activated', None)
            if callable(get_act):
                act = get_act()
                accel = ','.join(act) if isinstance(act, (list, tuple)) else str(act)
        except Exception:
            pass
        info['cupy_accel_active'] = accel
        # cuda versions
        try:
            info['cuda_runtime_ver'] = cp.cuda.runtime.runtimeGetVersion()
        except Exception:
            pass
        try:
            info['cuda_driver_ver'] = cp.cuda.driver.get_version()
        except Exception:
            pass
        # device
        try:
            dev = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get('name', 'unknown')
            if isinstance(name, bytes):
                name = name.decode()
            info['device_name'] = name
            cc = (props.get('major', None), props.get('minor', None))
            info['device_cc'] = cc
            info['mem_info'] = dev.mem_info  # (free,total)
        except Exception:
            pass
        # cutensor availability (robust across versions)
        cutensor_ok = None
        try:
            from cupy.cuda import cutensor as cu_ct  # CuPy ≥ 11-ish
            cutensor_ok = bool(getattr(cu_ct, 'available', False) or getattr(cu_ct, 'is_available', lambda: False)())
        except Exception:
            try:
                from cupy_backends.cuda.libs import cutensor as cu_ct2  # older
                cutensor_ok = bool(getattr(cu_ct2, 'is_available', lambda: False)())
            except Exception:
                pass
        if cutensor_ok is not True:
            try:
                import nvidia.cutensor as nct  # type: ignore
                cutensor_ok = True
                info['cutensor_pkg_ver'] = getattr(nct, '__version__', None)
            except Exception:
                pass
        info['cutensor_available'] = cutensor_ok
        # show_config (trimmed)
        try:
            from io import StringIO
            sio = StringIO()
            cp.show_config(stream=sio)
            text = sio.getvalue()
            keep = []
            for ln in text.splitlines():
                if any(k in ln for k in ('CUDA Build Version', 'cuTENSOR', 'cuBLAS', 'cuDNN')):
                    keep.append(ln)
            info['cupy_show_config'] = '\n'.join(keep) if keep else text
        except Exception:
            pass
    except Exception:
        pass
    return info


def print_snapshot(info: Dict[str, object]) -> None:
    hdr('Environment')
    print('python:', info['python'], '| exe:', info['exe'])
    for k, v in (info['env'] or {}).items():
        print(f'{k}:', v if v is not None else '(unset)')
    hdr('Modules / Versions / Paths')
    print('PySCF:', info['pyscf_ver'], '|', info['pyscf_path'])
    print('gpu4pyscf:', info['gpu4pyscf_ver'], '|', info['gpu4pyscf_path'])
    print('CuPy:', info['cupy_ver'], '|', info['cupy_path'])
    hdr('CUDA Device')
    print('runtime_ver:', info['cuda_runtime_ver'], 'driver_ver:', info['cuda_driver_ver'])
    print('device:', info['device_name'], 'compute_capability:', info['device_cc'])
    print('mem (free,total):', info['mem_info'])
    hdr('Accelerators')
    print('CUPY_ACCELERATORS(active):', info['cupy_accel_active'])
    print('gpu4pyscf.engine:', info['gpu4pyscf_engine'])
    print('cuTENSOR available:', info['cutensor_available'], 'nvidia-cutensor ver:', info['cutensor_pkg_ver'])
    sc = info.get('cupy_show_config')
    if sc:
        hdr('cupy.show_config (trim)')
        print(sc)


# ---------- NAO estimate from PDB ----------

def estimate_nao_from_pdb(pdb_path: str, basis: str) -> Optional[int]:
    """
    Minimal PDB read. Element from cols 77-78 if present; else first letter of atom name.
    Returns mol.nao_nr() for the chosen basis; None on failure.
    """
    try:
        atoms: List[str] = []
        with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for ln in fh:
                if not (ln.startswith('ATOM') or ln.startswith('HETATM')): continue
                # PDB element field
                el = ln[76:78].strip() if len(ln) >= 78 else ''
                if not el:
                    el = ln[12:16].strip()
                el = ''.join([ch for ch in el if ch.isalpha()]).capitalize()[:2] or 'C'
                try:
                    x = float(ln[30:38]);
                    y = float(ln[38:46]);
                    z = float(ln[46:54])
                except Exception:
                    continue
                atoms.append(f"{el} {x} {y} {z}")
        if not atoms:
            return None
        from pyscf import gto  # import here to avoid top-level import if PySCF missing
        mol = gto.M(atom='; '.join(atoms), unit='Angstrom', basis=basis, verbose=0)
        return int(mol.nao_nr())
    except Exception:
        return None


def recommend_policy(nao: Optional[int], accel_active: str) -> str:
    cutensor_on = 'cutensor' in (accel_active or '').lower()
    if nao is None:
        return 'recommendation: enable cuTENSOR; if unknown NAO, keep GPU for small systems only'
    if cutensor_on:
        return f"recommendation: GPU OK up to ~3000 NAO; current NAO={nao}"
    else:
        return f"recommendation: CPU if NAO>~800 (cupy-only); current NAO={nao}"


# ---------- subprocess SCF probe ----------

def _spawn(code: str, env: Optional[Dict[str, str]] = None, timeout: int = 60) -> Tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, '-c', code],
        env={**os.environ, **(env or {})},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout
    )
    return (proc.returncode, proc.stdout, proc.stderr)


def probe_gpu_scf(basis: str, xc: str, cycles: int, stress: int,
                  accel_env: Optional[str], timeout: int) -> Tuple[int, str, str]:
    """
    Child process: build small mol (H2O) or stress-sized CH4 grid; run gpu4pyscf RKS.
    Prints to_gpu/kernel status and NAO; rc encodes status (0=OK, 3/4=fails, 139=segfault).
    """
    mol_builder = """
from pyscf import gto
def make_mol(basis, stress):
    if stress <= 0:
        atom = "O 0 0 0; H 0 0 0.96; H 0.92 0 0"
    else:
        pts = []
        step = 2.2
        import math
        n = int(math.ceil(stress ** (1/3)))
        count = 0
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    if count >= stress: break
                    x, y, z = ix*step, iy*step, iz*step
                    pts.append(f"C {x} {y} {z}")
                    pts += [f"H {x+0.63} {y+0.63} {z+0.63}",
                            f"H {x-0.63} {y-0.63} {z+0.63}",
                            f"H {x-0.63} {y+0.63} {z-0.63}",
                            f"H {x+0.63} {y-0.63} {z-0.63}"]
                    count += 1
                if count >= stress: break
            if count >= stress: break
        atom = "; ".join(pts)
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    return mol
"""
    code = textwrap.dedent(f"""
import os, sys
os.environ.setdefault("PYSCF_MAX_MEMORY", "4000")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pyscf import dft
try:
    from gpu4pyscf import dft as g4dft
except Exception as e:
    print("IMPORT gpu4pyscf FAILED:", e)
    sys.exit(2)
{mol_builder}
basis={basis!r}; xc={xc!r}; stress={int(stress)}
mol = make_mol(basis, stress)
nao = mol.nao_nr()
mf = g4dft.RKS(mol)
mf.xc = xc
mf.max_cycle = {int(cycles)}
mf.conv_tol = 1e-7
try:
    mf = mf.to_gpu()
    print("to_gpu OK | nao=", nao)
except Exception as e:
    print("to_gpu FAILED:", repr(e), "| nao=", nao)
    sys.exit(3)
try:
    e = mf.kernel()
    print("kernel OK | E(Ha)=", e, "| nao=", nao)
except Exception as e:
    print("kernel FAILED:", repr(e), "| nao=", nao)
    sys.exit(4)
sys.exit(0)
""")
    env = dict(os.environ)
    if accel_env is not None:
        env['CUPY_ACCELERATORS'] = accel_env
    return _spawn(code, env=env, timeout=timeout)


# ---------- CLI ----------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Diagnose CuPy/cuTENSOR/gpu4pyscf SCF stability (segfault-safe).')
    ap.add_argument('--pdb', type=str, default=None, help='Optional PDB to estimate NAO (e.g., strychnine cluster)')
    ap.add_argument('--basis', type=str, default='def2-svp', help='Basis for NAO estimate and probes')
    ap.add_argument('--xc', type=str, default='b3lyp', help='XC functional for probes')
    ap.add_argument('--try-accel', action='append', default=[], help='CUPY_ACCELERATORS variants to try; repeatable')
    ap.add_argument('--stress', type=int, default=0, help='Replicate CH4 units (approx size control), e.g., 24')
    ap.add_argument('--cycles', type=int, default=3, help='SCF cycles for probe')
    ap.add_argument('--timeout', type=int, default=60, help='Per-probe timeout (s)')
    ap.add_argument('--escalate', action='store_true', help='Sweep stress levels until failure')
    args = ap.parse_args(argv)

    info = snapshot_runtime()
    print_snapshot(info)

    nao = None
    if args.pdb and os.path.exists(args.pdb):
        hdr('AO size estimate from PDB')
        nao = estimate_nao_from_pdb(args.pdb, args.basis)
        print('pdb:', args.pdb, 'basis:', args.basis, 'nao:', nao)
        print(recommend_policy(nao, str(info.get('cupy_accel_active') or '')))

    accel_tests = args.try_accel or [os.environ.get('CUPY_ACCELERATORS', None)]
    hdr('GPU SCF probes (subprocess)')

    def run_probe(stress: int, accel: Optional[str]) -> Tuple[int, str, str]:
        label = accel if accel is not None else '(inherit)'
        print(
            f'\n-- Probe: CUPY_ACCELERATORS={label} | basis={args.basis} | xc={args.xc} | stress={stress} | cycles={args.cycles}')
        try:
            rc, out, err = probe_gpu_scf(
                basis=args.basis, xc=args.xc, cycles=args.cycles,
                stress=stress, accel_env=accel, timeout=args.timeout
            )
            status = 'OK' if rc == 0 else ('SEGFAULT' if rc == 139 else f'FAIL(rc={rc})')
            print('status:', status)
            if out.strip(): print(out.strip())
            if err.strip(): print('[stderr]', err.strip())
            return (rc, out, err)
        except subprocess.TimeoutExpired:
            print('status: TIMEOUT')
            return (-1, '', 'timeout')
        except Exception as e:
            print('status: ERROR', repr(e))
            return (-2, '', repr(e))

    # single shot or escalation sweep
    stresses = [args.stress] if not args.escalate else [0, 1, 2, 4, 8, 12, 16, 24, 36, 48, 64]
    for accel in accel_tests:
        for s in stresses:
            rc, _, _ = run_probe(s, accel)
            if args.escalate and rc not in (0,):  # stop on first failure in sweep
                print(f'break: first failure at stress={s} (rc={rc}) for accelerators={accel}')
                break

    hdr('Next steps')
    print('* If status shows SEGFAULT or FAIL for large stress or your PDB NAO:')
    print('  - Install CUDA-matched CuPy and cuTENSOR; then:  export CUPY_ACCELERATORS=cub,cutensor')
    print('    pip:   pip install -U "cupy-cuda12x" "nvidia-cutensor-cu12"')
    print('    conda: conda install -c conda-forge cupy cutensor')
    print('* If cutensor remains inactive, purge duplicate CuPy builds; check LD_LIBRARY_PATH vs CONDA_PREFIX.')
    print('* Until cuTENSOR is active, gate GPU by NAO (~≤800 cupy-only) or force CPU for big systems.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
