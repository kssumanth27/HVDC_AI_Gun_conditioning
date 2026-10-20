"""
polgun_correlations.py
======================

Lagged Pearson correlation between the diagnostic signal pairs
(Current-Pressure, Current-Radiation, Pressure-Radiation) across the
training conditioning runs (manuscript Fig. 2).

For each training file the full r(tau) curve is computed for lags
-MAX_LAG..+MAX_LAG seconds. The curves are then averaged (mean and median)
across files lag by lag, and on each aggregate curve the script reports:
    tau_p  : lag of the peak |r|
    peak r : r at tau_p
    FWHM   : width of the peak at half its height (linear interpolation)

Pearson r is invariant to z-normalization, so raw signals are used.

Outputs (in OUTPUT_DIR):
    pearson_{mean|median}_{pair}_{timestamp}_{DPI}dpi.png   one plot per pair/aggregate
    pearson_lag_summary_{timestamp}.csv                       tau_p, peak r, FWHM per pair/aggregate

Usage
-----
    python polgun_correlations.py
"""

import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr


# ============================ CONFIGURATION ============================
TRAIN_DIR = (
    r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2"
    r"\polgun v8 until max conditioning\v8 spikes cleaned until max\train poster"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crosscorr_figures")

SENSOR_COLS = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
PAIR_NAMES = ["Current-Pressure", "Current-Radiation", "Pressure-Radiation"]
PAIR_INDICES = [(0, 1), (0, 2), (1, 2)]
MAX_LAG = 50
LAG_STEP_SECONDS = 1.0
DPI = 1200
FWHM_FACE = "#F4C27A"
FWHM_ALPHA = 0.28
# =======================================================================


def load_sensor_files(data_dir):
    """Read the sensor columns of every CSV that contains all three of them."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    file_dfs = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=lambda c: c in SENSOR_COLS)
        if len(df) and all(c in df.columns for c in SENSOR_COLS):
            for c in SENSOR_COLS:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            file_dfs.append(df)
    return file_dfs


def lagged_pearson(a, b, max_lag=MAX_LAG):
    """Pearson r between a and b for every lag in [-max_lag, max_lag].

    Positive lag pairs a[t + lag] with b[t]. Returns None if the series is
    too short for the lag range.
    """
    a = np.asarray(pd.to_numeric(a, errors="coerce").dropna(), dtype=float)
    b = np.asarray(pd.to_numeric(b, errors="coerce").dropna(), dtype=float)
    n = min(len(a), len(b))
    if n < 2 * max_lag + 10:
        return None
    a, b = a[:n], b[:n]
    out = np.full(2 * max_lag + 1, np.nan)
    for k, lag in enumerate(np.arange(-max_lag, max_lag + 1)):
        if lag >= 0:
            x, y = a[lag:], b[: n - lag]
        else:
            x, y = a[: n + lag], b[-lag:]
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        out[k], _ = pearsonr(x, y)
    return out


def fwhm_bounds(curve, lags_arr):
    """Return (peak_lag, peak_r, fwhm, left, right) of the peak |r| of ``curve``.

    The half-maximum crossings on either side of the peak are found by linear
    interpolation between neighbouring lags.
    """
    curve = np.asarray(curve, float)
    peak_idx = int(np.nanargmax(np.abs(curve)))
    peak_val = float(curve[peak_idx])
    half = peak_val / 2.0
    left = None
    for i in range(peak_idx, 0, -1):
        y0, y1 = curve[i - 1], curve[i]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if (y0 - half) * (y1 - half) <= 0:
            left = float(
                lags_arr[i - 1]
                + (half - y0) * (lags_arr[i] - lags_arr[i - 1]) / (y1 - y0)
            ) if y1 != y0 else float(lags_arr[i])
            break
    right = None
    for i in range(peak_idx, len(curve) - 1):
        y0, y1 = curve[i], curve[i + 1]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if (y0 - half) * (y1 - half) <= 0:
            right = float(
                lags_arr[i]
                + (half - y0) * (lags_arr[i + 1] - lags_arr[i]) / (y1 - y0)
            ) if y1 != y0 else float(lags_arr[i])
            break
    fwhm = float(right - left) if left is not None and right is not None else np.nan
    return float(lags_arr[peak_idx]), peak_val, fwhm, left, right


def compute_aggregate_curves(file_dfs):
    """Per-file r(tau) curves per pair, aggregated across files by median and mean."""
    all_curves = {pn: [] for pn in PAIR_NAMES}
    for df in file_dfs:
        for (i, j), pn in zip(PAIR_INDICES, PAIR_NAMES):
            c = lagged_pearson(df[SENSOR_COLS[i]], df[SENSOR_COLS[j]])
            if c is not None:
                all_curves[pn].append(c)

    agg_curves = {}
    for pn in PAIR_NAMES:
        mat = np.vstack(all_curves[pn])
        agg_curves[(pn, "median")] = np.nanmedian(mat, axis=0)
        agg_curves[(pn, "mean")] = np.nanmean(mat, axis=0)
    return agg_curves


def plot_lagged_correlation(rcurve, lag_seconds, peak_lag_s, fwhm, left, right):
    """Bar plot of r(tau) with the tau_p line and the shaded FWHM band."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    if left is not None and right is not None and np.isfinite(fwhm):
        ax.axvspan(left, right, facecolor=FWHM_FACE, alpha=FWHM_ALPHA, zorder=0, linewidth=0)

    ax.bar(
        lag_seconds,
        rcurve,
        width=LAG_STEP_SECONDS * 0.9,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.85,
        zorder=2,
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=3)
    ax.axvline(peak_lag_s, color="red", linestyle="-", linewidth=1.5, zorder=4)

    fwhm_str = (
        rf"$\mathrm{{FWHM}} = {fwhm:.2f}\ \mathrm{{s}}$"
        if np.isfinite(fwhm)
        else r"$\mathrm{FWHM} = \mathrm{n/a}$"
    )
    legend_handles = [
        plt.Line2D([0], [0], color="red", lw=1.5,
                   label=rf"$\tau_p = {peak_lag_s:.3f}\ \mathrm{{s}}$"),
        Patch(facecolor=FWHM_FACE, edgecolor="none",
              alpha=min(FWHM_ALPHA + 0.25, 0.7), label=fwhm_str),
    ]
    ax.legend(handles=legend_handles, frameon=True, loc="best")

    ax.set_xlabel("Time Lag (s)")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25, linewidth=0.6, zorder=1)
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_dfs = load_sensor_files(TRAIN_DIR)
    print(f"Loaded {len(file_dfs)} files  stamp={stamp}  dpi={DPI}")
    if not file_dfs:
        raise SystemExit("No valid CSV files found.")

    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    lag_seconds = lags * LAG_STEP_SECONDS
    agg_curves = compute_aggregate_curves(file_dfs)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "axes.linewidth": 1.0,
    })

    summary_rows = []
    for pn in PAIR_NAMES:
        for agg in ("median", "mean"):
            rcurve = agg_curves[(pn, agg)]
            peak_lag_s, peak_r, fwhm, left, right = fwhm_bounds(rcurve, lag_seconds)

            fig = plot_lagged_correlation(rcurve, lag_seconds, peak_lag_s, fwhm, left, right)
            fname = f"pearson_{agg}_{pn.replace('-', '_')}_{stamp}_{DPI}dpi.png"
            path = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            print(f"{agg:6s} {pn:20s}  tau_p={peak_lag_s:.3f}s  r={peak_r:.4f}  "
                  f"FWHM={fwhm:.2f}s  -> {fname}")
            summary_rows.append({
                "pair": pn,
                "aggregate": agg,
                "tau_p_s": peak_lag_s,
                "peak_r": peak_r,
                "lag0_r": float(rcurve[MAX_LAG]),
                "fwhm_s": fwhm,
                "fwhm_left_s": left,
                "fwhm_right_s": right,
                "n_files": len(file_dfs),
            })

    summary_path = os.path.join(OUTPUT_DIR, f"pearson_lag_summary_{stamp}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
