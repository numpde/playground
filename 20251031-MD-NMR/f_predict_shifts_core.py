#!/usr/bin/env python3
# f_predict_shifts_core.py
#
# Core utilities for NMR shift / J coupling / Boltzmann weighting pipeline.
#
# Imported by:
#   - f_predict_shifts_compute.py   (per-cluster quantum calcs)
#   - f_predict_shifts_average.py   (Boltzmann averaging / fast exchange)

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyscf import gto, dft, scf

# Silence PySCF "under testing" noise, keep real errors visible
warnings.filterwarnings(
    "ignore",
    message=r"Module .* is under testing",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Module .* is not fully tested",
    category=UserWarning,
)

LOG = logging.getLogger("nmrshifts.core")

# ---- Optional GPU backend -------------------------------------------------
try:
    from gpu4pyscf import dft as g4dft  # type: ignore

    GPU4PYSCF_AVAILABLE = True
except Exception:
    GPU4PYSCF_AVAILABLE = False

# ---- Optional Berny (geometry optimizer) ----------------------------------
try:
    import berny  # type: ignore

    BERNY_AVAILABLE = True
except Exception:
    BERNY_AVAILABLE = False

# ---- Paths / defaults -----------------------------------------------------
OUT_DIR = Path("f_predict_shifts")
CLUSTERS_DIR = Path("e_cluster")
DFT_XC_DEFAULT = "b3lyp"
BASIS_DEFAULT = "def2-tzvp"
SCF_MAXCYC = 200
SCF_CONV_TOL = 1e-9
GRAD_CONV_TOL = 3e-4  # reserved for future geom opt

# ---- Solvent map (eps ~298 K). Override with --eps ------------------------
SOLVENT_EPS: Dict[str, float] = {
    "vacuum": 1.0,
    "water": 78.36,
    "h2o": 78.36,
    "dmso": 46.7,
    "meoh": 32.7,
    "methanol": 32.7,
    "mecn": 35.7,
    "acetonitrile": 35.7,
    "chcl3": 4.81,
    "cdcl3": 4.81,
    "chloroform": 4.81,
    "thf": 7.58,
    "toluene": 2.38,
    "acetone": 20.7,
}


# -----------------------------------------------------------------------------
# Logging / convenience
# -----------------------------------------------------------------------------

@dataclass
class ClusterRow:
    cid: int
    fraction: float  # MD or clustering population fraction
    rep_pdb: Path  # representative conformer geometry for this cluster


def setup_logging(level: str = "INFO", quiet: bool = False) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    if quiet:
        numeric = max(numeric, logging.WARNING)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pyscf").setLevel(logging.WARNING)


def detect_gpu() -> Dict[str, object]:
    """
    Report whether gpu4pyscf / CuPy devices are visible.
    Used only for logging + --gpu auto heuristics.
    """
    info = {"gpu4pyscf": GPU4PYSCF_AVAILABLE, "cupy": False, "devices": []}
    try:
        import cupy as cp  # type: ignore

        n = int(cp.cuda.runtime.getDeviceCount())
        info["cupy"] = n > 0
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            info["devices"].append(str(name))
    except Exception:
        pass
    return info


def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fmt(x: float) -> str:
    return f"{x:.6f}" if np.isfinite(x) else "nan"


# -----------------------------------------------------------------------------
# Cluster table / geometry I/O
# -----------------------------------------------------------------------------

def get_charge_spin(tag: str) -> Tuple[int, int]:
    """
    Heuristic net charge / spin multiplicity guess from tag name.
    If tag says 'deprot' or 'anion' -> -1 charge.
    If tag says 'prot' or 'cation'  -> +1 charge.
    Otherwise neutral singlet.
    """
    t = tag.lower()
    if "deprot" in t or "anion" in t:
        return (-1, 0)
    if "cation" in t or "prot" in t:
        return (+1, 0)
    return (0, 0)


def guess_rep_path(tag: str, cid: int) -> Path:
    """
    Try common filename patterns to locate cluster representative PDB.
    Falls back to <tag>_cluster_<cid>_rep.pdb even if it doesn't exist yet.
    """
    p = CLUSTERS_DIR / f"{tag}_cluster_{cid}_rep.pdb"
    if p.exists():
        return p
    for name in (
        f"{tag}_cluster{cid}_rep.pdb",
        f"{tag}_c{cid}_rep.pdb",
        f"{tag}_cluster_{cid}_representative.pdb",
        f"{tag}_cluster_{cid}.pdb",
    ):
        q = CLUSTERS_DIR / name
        if q.exists():
            return q
    return p


def load_clusters_table(path: Path, tag: str) -> List[ClusterRow]:
    """
    Load cluster summary for `tag`.

    Supports two formats:

    A) Headered TSV (cid/fraction/rep_pdb/...); we fuzzy-match column names.

    B) Commented TSV that looks like:
       #cluster_id  count  fraction  medoid_frame_idx  rep_pdb_path
       0            123    0.2049    157               foo_cluster_0_rep.pdb
       ...

    Returns
    -------
    list[ClusterRow]
    """
    # --- attempt A: parse as dataframe with headers -------------------------
    try:
        df_try = pd.read_csv(path, sep=None, engine="python", comment="#").copy()
        cols = {c.lower(): c for c in df_try.columns}

        cid_col = None
        for cand in (
            "cid",
            "cluster",
            "cluster_id",
            "clusteridx",
            "cluster_idx",
            "id",
        ):
            if cand in cols:
                cid_col = cols[cand]
                break

        if cid_col is not None:
            # we think this has headers
            frac_col = None
            for cand in (
                "fraction",
                "pop",
                "population",
                "weight",
                "boltz",
                "md_fraction",
                "mdfrac",
                "md_weight",
            ):
                if cand in cols:
                    frac_col = cols[cand]
                    break

            rep_col = None
            for cand in (
                "rep",
                "rep_pdb",
                "rep_path",
                "pdb",
                "pdb_path",
                "representative",
            ):
                if cand in cols:
                    rep_col = cols[cand]
                    break

            cidv = pd.to_numeric(df_try[cid_col], errors="raise")

            if frac_col is not None:
                frv = pd.to_numeric(df_try[frac_col], errors="coerce")
            else:
                frv = None

            if rep_col is not None:
                rpv = df_try[rep_col].astype(str)
            else:
                rpv = None

            rows: List[ClusterRow] = []
            for i in range(len(df_try)):
                cid_i = int(cidv.iloc[i])

                # fraction
                if frv is not None and pd.notna(frv.iloc[i]):
                    frac_i = float(frv.iloc[i])
                else:
                    frac_i = float("nan")

                # representative pdb path
                if rpv is not None and rpv.iloc[i].strip():
                    rp_raw = Path(rpv.iloc[i].strip())
                    rep_path = (
                        rp_raw
                        if rp_raw.is_absolute()
                        else (path.parent / rp_raw)
                    )
                else:
                    rep_path = guess_rep_path(tag, cid_i)

                rows.append(
                    ClusterRow(
                        cid=cid_i,
                        fraction=frac_i,
                        rep_pdb=rep_path.resolve(),
                    )
                )

            # Fill NaN fractions uniformly
            fracs = np.array([r.fraction for r in rows], dtype=float)
            if not np.all(np.isfinite(fracs)):
                n = len(rows)
                for r in rows:
                    r.fraction = 1.0 / n

            return rows

    except Exception:
        # fall through to attempt B
        pass

    # --- attempt B: manual parse of commented TSV --------------------------
    rows = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expect 5+ cols:
            # cluster_id  count  fraction  medoid_frame_idx  rep_pdb_path
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"[{path}] can't parse line (need 5+ cols): {line!r}"
                )

            cid_i = int(parts[0])
            frac_i = float(parts[2])
            rep_rel = parts[4]
            rep_path = (path.parent / rep_rel).resolve()

            rows.append(
                ClusterRow(
                    cid=cid_i,
                    fraction=frac_i,
                    rep_pdb=rep_path,
                )
            )

    # Fill NaN fractions uniformly, if any
    fracs = np.array([r.fraction for r in rows], dtype=float)
    if not np.all(np.isfinite(fracs)):
        n = len(rows)
        for r in rows:
            r.fraction = 1.0 / n

    return rows


def read_pdb_atoms(pdb: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Minimal PDB reader: return
        atom_names:  ['H12', 'C
