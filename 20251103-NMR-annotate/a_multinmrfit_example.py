# a_multinmrfit_example.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import multinmrfit.base.io as _mnf_io
# MultiNMRFit API
import multinmrfit.base.spectrum as _mnf_spectrum
# pip install "git+https://github.com/NMRTeamTBI/MultiNMRFit.git"
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Annotation:
    signal_id: str
    model: str  # singlet | doublet | triplet | quartet
    n_peaks: int
    delta_ppm: float
    J_ppm: Optional[float]  # None for singlet; mean spacing for triplet/quartet
    J_Hz: Optional[float]  # converted using spectrometer_MHz
    lw_ppm: Optional[float]  # model linewidth parameter (≈ HWHM in ppm)
    lw_Hz: Optional[float]  # 2*lw_ppm*spectrometer_MHz (≈ FWHM in Hz)
    area: Optional[float]  # fitted integral (a.u.)


def _robust_threshold(y: np.ndarray) -> float:
    """
    Robust absolute threshold for peak_picking.
    - Trim top/bottom 1% to ignore solvent spikes.
    - Use 6*MAD as a noise-based cut.
    - If MAD==0 (flat baseline), fall back to 1% of max amplitude.
    """
    y = np.asarray(y, float).ravel()
    if y.size == 0:
        return 0.0
    ymax = float(np.nanmax(np.abs(y))) if np.isfinite(y).all() else 0.0
    if ymax <= 0.0:
        return 0.0
    (lo, hi) = np.quantile(y, [0.01, 0.99])
    core = y[(y >= lo) & (y <= hi)]
    if core.size < 8:
        return 0.01 * ymax
    mad = np.median(np.abs(core - np.median(core)))
    sigma = 1.4826 * mad
    thr = max(6.0 * sigma, 0.01 * ymax)
    return float(thr)


def _cluster_peaks(ppm: np.ndarray, eps_ppm: float) -> np.ndarray:
    # Single-dimension DBSCAN (no sklearn dep): greedy clustering on sorted coords
    order = np.argsort(ppm)
    labels = -np.ones_like(order)
    cid = 0
    last = None
    for i in order:
        if last is None or abs(ppm[i] - last) > eps_ppm:
            cid += 1
        labels[i] = cid
        last = ppm[i]
    return labels


def annotate_h1_spectrum_with_multinmrfit(
        ppm: np.ndarray,
        intensity: np.ndarray,
        *,
        spectrometer_MHz: float,
        window: Optional[Tuple[float, float]] = None,
        peak_threshold: Optional[float] = None,
        cluster_eps_Hz: float = 30.0,  # cluster lines within ~30 Hz by default
) -> pd.DataFrame:
    """
    End-to-end ^1H peak picking + multiplet fitting using MultiNMRFit.

    Inputs
    ------
    ppm, intensity : 1D arrays (same length), pre-phased & baseline-corrected (recommended).
    spectrometer_MHz : spectrometer frequency to convert J [ppm] -> [Hz].
    window : (min_ppm, max_ppm) to limit analysis; default = full range.
    peak_threshold : override automatic threshold for peak picking (in intensity units).
    cluster_eps_Hz : peaks within this spacing are grouped into one multiplet.

    Output
    ------
    pandas.DataFrame with columns:
      ['signal_id','model','n_peaks','delta_ppm','J_ppm','J_Hz','lw_ppm','lw_Hz','area'].
    """
    # 0) sanitize inputs
    assert ppm.ndim == 1 and intensity.ndim == 1 and ppm.size == intensity.size
    df = pd.DataFrame({'ppm': ppm.astype(float), 'intensity': intensity.astype(float)})

    # 1) build Spectrum (window optional; MultiNMRFit uses ppm ascending internally)
    if window is not None:
        wmin, wmax = float(min(window)), float(max(window))
        df = df[(df['ppm'] >= wmin) & (df['ppm'] <= wmax)].copy()
    df.sort_values('ppm', inplace=True, ignore_index=True)

    sp = _mnf_spectrum.Spectrum(data=df, window=None)  # full current df window

    # 2) automatic peak picking
    if peak_threshold is None:
        thr = _robust_threshold(df['intensity'].values)
        peaks = sp.peak_picking(threshold=thr if thr > 0 else None)
    else:
        peaks = sp.peak_picking(threshold=float(peak_threshold))

    if len(peaks) == 0:
        return pd.DataFrame(columns=[
            'signal_id', 'model', 'n_peaks', 'delta_ppm', 'J_ppm', 'J_Hz', 'lw_ppm', 'lw_Hz', 'area'
        ])

    # 3) cluster peaks into candidate multiplets
    eps_ppm = max(0.003, cluster_eps_Hz / spectrometer_MHz)  # ~>= 3 Hz safety floor
    labels = _cluster_peaks(peaks['ppm'].to_numpy(), eps_ppm)
    peaks = peaks.assign(cluster=labels)

    # 4) choose a model per cluster by #lines (Pascal-ratio heuristic for 4 lines)
    model_map: Dict[int, str] = {1: 'singlet', 2: 'doublet', 3: 'triplet', 4: 'quartet'}
    signals: Dict[str, Dict] = {}
    models = _mnf_io.IoHandler.get_models()  # includes singlet/doublet/triplet/quartet

    for k, sub in peaks.groupby('cluster', sort=True):
        sub = sub.sort_values('ppm')
        n = len(sub)
        if n not in model_map:
            # fallback: collapse to singlet at centroid if very complex
            model_name = 'singlet'
        else:
            model_name = model_map[n]
            if n == 4:
                # crude dd vs quartet decision: prefer quartet if close to 1:3:3:1
                v = sub['intensity'].to_numpy().astype(float)
                pattern = np.array([1, 3, 3, 1], float)
                if np.sum(v) > 0:
                    v = v / v.sum()
                    pattern = pattern / pattern.sum()
                if np.linalg.norm(v - pattern, 1) > 0.6:
                    # intensities far from Pascal; the shipped 'doublet of doublet' exists,
                    # but it requires a different parameterization—sticking to quartet keeps the API simple.
                    model_name = 'quartet'

        # parameter seeds
        x0 = float(sub['ppm'].mean())
        total_I = float(sub['intensity'].sum())
        span = float(sub['ppm'].max() - sub['ppm'].min())
        J_ppm = (span / (n - 1)) if n > 1 else None
        lw_ppm_ini = max(0.001, 0.25 * (span / max(1, n - 1))) if n > 1 else 0.003

        par: Dict[str, Dict[str, float]] = {
            'x0': {'ini': x0, 'lb': x0 - 0.1, 'ub': x0 + 0.1},
            'intensity': {'ini': max(total_I, 1.0), 'ub': max(10.0 * total_I, 1e3)},
            'lw': {'ini': lw_ppm_ini, 'lb': 0.0001, 'ub': 0.03},
            'gl': {'ini': 0.5, 'lb': 0.0, 'ub': 1.0},
        }
        if J_ppm is not None:
            par['J'] = {'ini': J_ppm, 'lb': 0.25 * J_ppm, 'ub': 4.0 * J_ppm}

        sid = f"sig_{int(k)}"
        signals[sid] = {'model': model_name, 'par': par}

    # 5) build + fit
    sp.build_model(signals=signals, available_models=models)
    sp.fit()  # L-BFGS-B by default

    # 6) collect fitted parameters per signal
    P = sp.params.copy()  # columns include: signal_id, model, par, opt, integral, ...
    # Keep one row per signal with the fields we want
    out: List[Annotation] = []
    for sid, rows in P.groupby('signal_id'):
        model = rows['model'].iloc[0]
        n = model_map.get({'singlet': 1, 'doublet': 2, 'triplet': 3, 'quartet': 4}.get(model, 1), 1)

        def _get(par: str, default=None):
            s = rows[rows['par'] == par]
            return float(s['opt'].iloc[0]) if len(s) else default

        x0 = _get('x0')
        Jppm = _get('J', None)
        lw = _get('lw', None)
        area = rows['integral'].dropna().iloc[0] if rows['integral'].notna().any() else None
        out.append(Annotation(
            signal_id=str(sid),
            model=str(model),
            n_peaks=n,
            delta_ppm=x0,
            J_ppm=Jppm,
            J_Hz=(abs(Jppm) * spectrometer_MHz if Jppm is not None else None),
            lw_ppm=lw,
            lw_Hz=(2.0 * abs(lw) * spectrometer_MHz if lw is not None else None),  # ≈ FWHM
            area=area,
        ))

    return pd.DataFrame([a.__dict__ for a in out]).sort_values('delta_ppm', ascending=False).reset_index(drop=True)


def main():
    src = "/home/ra/Datasets/2024-Alberts/b_extracted/multimodal_spectroscopic_dataset/aligned_chunk_0.parquet"
    smiles_col = 'smiles'
    h_nmr_sig_col = 'h_nmr_spectra'
    h_nmr_ann_col = 'h_nmr_peaks'

    # print(list(pq.read_schema(src).names))

    cols = [smiles_col, h_nmr_sig_col, h_nmr_ann_col]

    # 3) Read just what we need
    df = pq.read_table(src, columns=cols).to_pandas()

    # 4) Take the first row that actually contains a spectrum
    df = df.dropna(subset=[h_nmr_sig_col])

    assert not df.empty

    row = df.iloc[0]

    print(row[smiles_col])

    y = row[h_nmr_sig_col]
    y = np.asarray(y, dtype=float).ravel()

    # 5) Spectrometer frequency (Hz conversion for J). Default conservatively to 400 MHz.
    spectrometer_MHz = 600.0

    # 6) Build a ppm axis matching y (ascending; annotate() sorts internally anyway)
    #    Range is a placeholder; tighten if your dataset encodes ppm elsewhere.
    # As per `spectrum_dimensions.json` at
    # https://github.com/rxn4chemistry/multimodal-spectroscopic-dataset/blob/5d6934f3588087e5829b0acaa28fae4a01f216ce/data/meta_data/spectrum_dimensions.json
    x_ppm = np.linspace(10, -2, y.size, dtype=float)

    from plox import Plox
    from bugs import mkdir
    with Plox() as px:
        px.a.plot(x_ppm, y)
        px.a.invert_xaxis()
        px.f.savefig(mkdir(Path(__file__).with_suffix('')) / f"spectrum_example.png")

    # invert x_ppm and y:
    x_ppm = x_ppm[::-1]
    y = y[::-1]

    # 7) Run end-to-end annotation (array → peak-pick → cluster → fit → shift/J/multiplet)
    annotations = annotate_h1_spectrum_with_multinmrfit(
        ppm=x_ppm,
        intensity=y,
        spectrometer_MHz=spectrometer_MHz,
        window=None,
        peak_threshold=None,  # auto via robust MAD
        cluster_eps_Hz=10.0,  # tune if over/under-clustering
    )

    # 8) Show results
    if annotations.empty:
        print("No peaks detected in the selected window.")
    else:
        # highest ppm first (conventional listing)
        print(annotations.to_string(index=False))

    print("Expected: ", row[h_nmr_ann_col])


if __name__ == "__main__":
    main()
