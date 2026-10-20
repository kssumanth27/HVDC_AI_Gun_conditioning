"""
PolarGun post-dataconvert pipeline.

Runs, per CSV produced by dataconvert.py:
  1. Spike clamp + bidirectional trend smoothing of the voltage
     (same logic as datalabelv4_smoothing 1.py)
  2. VoltageChange and RadiationTotal labelling
  3. Truncation to (first max voltage + n_after rows)
     (same logic as Data Label until max after cleaning PolarGun 1.py)

Only the columns needed downstream are written; *_missing and the
clamp/trend/freeze audit flags are not.
"""
import os
from glob import glob

import numpy as np
import pandas as pd


VOLT_COL = 'glassmanDataXfer:hvPsVoltageMeasM'

EXPECTED_COLUMNS = [
    'Time',
    VOLT_COL,
    'gc1_diff', 'gc2_diff', 'gc3_diff', 'gc4_diff',
    'GunCurrent.Avg',
    'peg-BL-cc:pressureM'
]

OUTPUT_COLUMNS = EXPECTED_COLUMNS + ['VoltageChange', 'RadiationTotal']

# ---- Config: hard cap for upward change in voltage ----
MAX_UP_STEP = 1300.0

PERSIST_LOOKAHEAD = 2       # require next 2 points to match the new high value
REL_TOL = 0.00              # relative tolerance to decide "matches the high value"
ABS_TOL = 500               # absolute tolerance buffer

# Trend-confirmation smoothing
TREND_LOOKAHEAD     = 5     # look ahead 5 points after a candidate change
TREND_REL_TOL       = 0.00
TREND_ABS_TOL       = 0.0
MIN_MEANINGFUL_STEP = 0.0

# Big decrease bypass (do not smooth next few points if drop is large)
BIG_DROP_MAG          = 3000.0
FREEZE_AFTER_BIG_DROP = 5

# Until-max truncation
N_AFTER_MAX = 10


def _matches(a, b, rel=REL_TOL, abs_=ABS_TOL):
    """Return True if a ~= b within relative/absolute tolerance."""
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(b), 1e-12))


def clamp_with_persistent_high(series: pd.Series,
                               max_up_step: float = MAX_UP_STEP,
                               persist: int = PERSIST_LOOKAHEAD) -> tuple[pd.Series, pd.Series]:
    """
    Forward pass that clamps large upward jumps, but allows a regime change
    if the next 'persist' raw points match the new high value.

    Algo:
      - If (raw[i] - fixed[i-1]) > max_up_step => clamp fixed[i] = fixed[i-1].
      - On the *next* index j, if (raw[j] - fixed[j-1]) > max_up_step:
          look ahead raw[j+1..j+persist]; if all match raw[j] within tolerance,
          accept the jump (fixed[j] = raw[j]); else clamp again.
      - Otherwise fixed[i] = raw[i].
    Returns (fixed_series, clamped_flags_int).
    """
    x_raw = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    n = len(x_raw)
    if n == 0:
        return series.copy(), pd.Series([], dtype=int, index=series.index)

    x_fix = x_raw.copy()
    clamp_flags = np.zeros(n, dtype=int)

    prev_forced = False   # whether previous step was clamped

    i = 1
    while i < n:
        prev = x_fix[i-1]
        cand = x_raw[i]
        if np.isnan(prev):
            x_fix[i] = cand
            prev_forced = False
            i += 1
            continue

        if np.isnan(cand):
            x_fix[i] = cand
            prev_forced = False
            i += 1
            continue

        delta = cand - prev
        if delta > max_up_step:
            if prev_forced:
                ok = True
                if i + persist >= n:
                    ok = False  # not enough lookahead -> conservative: clamp
                else:
                    for k in range(1, persist + 1):
                        nxt = x_raw[i + k]
                        if not _matches(nxt, cand):
                            ok = False
                            break
                if ok:
                    x_fix[i] = cand
                    prev_forced = False
                else:
                    x_fix[i] = prev
                    clamp_flags[i] = 1
                    prev_forced = True
            else:
                x_fix[i] = prev
                clamp_flags[i] = 1
                prev_forced = True
        else:
            x_fix[i] = cand
            prev_forced = False

        i += 1

    return pd.Series(x_fix, index=series.index), pd.Series(clamp_flags, index=series.index)


def trend_confirm_bidir(series,
                        lookahead=TREND_LOOKAHEAD,
                        rel_tol=TREND_REL_TOL,
                        abs_tol=TREND_ABS_TOL,
                        min_step=MIN_MEANINGFUL_STEP,
                        big_drop_mag=BIG_DROP_MAG,
                        freeze_steps=FREEZE_AFTER_BIG_DROP):
    """
    Bidirectional jitter smoother with BIG DROP BYPASS:
      - Small changes (|cand - prev| <= eps(prev)) -> hold constant (prev).
      - Increase candidate (cand > prev + eps(prev)):
          Accept only if future 'lookahead' raw points NEVER drop significantly below cand,
          else hold (prev).
      - Decrease candidate (cand < prev - eps(prev)):
          If (prev - cand) > big_drop_mag:
              Accept cand, then BYPASS smoothing for the next `freeze_steps` points.
          Else accept only if future 'lookahead' raw points NEVER rise significantly above cand,
              else hold (prev).
    Returns (fixed_series, hold_inc_flags_int, hold_dec_flags_int, big_drop_freeze_start_flags_int).
    """
    raw = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    n = len(raw)
    if n == 0:
        return (series.copy(),
                pd.Series([], dtype=int, index=series.index),
                pd.Series([], dtype=int, index=series.index),
                pd.Series([], dtype=int, index=series.index))

    fix = raw.copy()
    hold_inc = np.zeros(n, dtype=int)
    hold_dec = np.zeros(n, dtype=int)
    big_drop_freeze = np.zeros(n, dtype=int)

    def eps(ref):
        return max(abs_tol, rel_tol * max(abs(ref), 1e-12), min_step)

    freeze = 0  # when >0, bypass trend smoothing (accept raw), then decrement each step

    for i in range(1, n):
        prev = fix[i-1]
        cand = raw[i]

        if freeze > 0:
            fix[i] = cand
            freeze -= 1
            continue

        if np.isnan(prev) or np.isnan(cand):
            fix[i] = cand
            continue

        delta = cand - prev
        e_prev = eps(prev)

        if -e_prev <= delta <= e_prev:
            fix[i] = prev
            continue

        j_end = min(n - 1, i + lookahead)
        if j_end <= i:
            fix[i] = prev
            if delta > 0: hold_inc[i] = 1
            else:         hold_dec[i] = 1
            continue

        e_cand = eps(cand)

        if delta > e_prev:
            reject = False
            for j in range(i + 1, j_end + 1):
                nxt = raw[j]
                if not np.isnan(nxt) and (cand - nxt) > e_cand:
                    reject = True; break
            if reject:
                fix[i] = prev; hold_inc[i] = 1
            else:
                fix[i] = cand
        else:
            drop_mag = prev - cand
            if drop_mag > big_drop_mag:
                fix[i] = cand
                big_drop_freeze[i] = 1
                freeze = freeze_steps
            else:
                reject = False
                for j in range(i + 1, j_end + 1):
                    nxt = raw[j]
                    if not np.isnan(nxt) and (nxt - cand) > e_cand:
                        reject = True; break
                if reject:
                    fix[i] = prev; hold_dec[i] = 1
                else:
                    fix[i] = cand

    return (pd.Series(fix, index=series.index),
            pd.Series(hold_inc, index=series.index),
            pd.Series(hold_dec, index=series.index),
            pd.Series(big_drop_freeze, index=series.index))


def label_and_smooth(df: pd.DataFrame) -> pd.DataFrame:
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time').drop_duplicates(subset='Time', keep='first').reset_index(drop=True)

    # Pass 1: big upward clamp with persistence check
    v_after_big, _ = clamp_with_persistent_high(df[VOLT_COL], MAX_UP_STEP, PERSIST_LOOKAHEAD)

    # Pass 2: bidirectional trend confirmation with big drop bypass
    v_after_trend, _, _, _ = trend_confirm_bidir(
        v_after_big,
        TREND_LOOKAHEAD, TREND_REL_TOL, TREND_ABS_TOL, MIN_MEANINGFUL_STEP,
        BIG_DROP_MAG, FREEZE_AFTER_BIG_DROP
    )
    df[VOLT_COL] = v_after_trend

    # VoltageChange: sign of NEXT - CURRENT voltage (1/0/-1); last row -> NaN
    diff_v = pd.to_numeric(df[VOLT_COL], errors='coerce').shift(-1) - pd.to_numeric(df[VOLT_COL], errors='coerce')
    vc = np.where(diff_v > 0, 1, np.where(diff_v < 0, -1, 0)).astype('float')
    vc[np.isnan(diff_v.values)] = np.nan
    df['VoltageChange'] = vc

    df['RadiationTotal'] = pd.concat(
        [pd.to_numeric(df[c], errors='coerce') for c in ['gc1_diff', 'gc2_diff', 'gc3_diff', 'gc4_diff']],
        axis=1
    ).sum(axis=1, min_count=1)

    return df


def select_upto_max_plus_n(df: pd.DataFrame, col: str = VOLT_COL, n_after: int = N_AFTER_MAX) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    s = pd.to_numeric(df[col], errors="coerce")
    arr = s.to_numpy()

    if np.isnan(arr).all():
        return df.iloc[0:0].copy()

    max_pos = int(np.nanargmax(arr))  # positional index of first max
    end_pos = min(max_pos + n_after, len(df) - 1)

    return df.iloc[:end_pos + 1].copy()


def process_one_file(in_path: str, out_path: str, n_after: int = N_AFTER_MAX):
    df = pd.read_csv(in_path)
    df = label_and_smooth(df)
    df = select_upto_max_plus_n(df, col=VOLT_COL, n_after=n_after)
    df = df[OUTPUT_COLUMNS]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)


def process_folder(input_dir: str, output_dir: str, n_after: int = N_AFTER_MAX):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted([p for p in glob(os.path.join(input_dir, "*")) if p.lower().endswith(".csv")])
    print(f"Found {len(paths)} CSV files in {input_dir}")

    for in_path in paths:
        base = os.path.basename(in_path)
        name, _ = os.path.splitext(base)
        out_filename = f"processed_{name}_uptoMax_plus{n_after}.csv"
        out_path = os.path.join(output_dir, out_filename)

        try:
            process_one_file(in_path, out_path, n_after=n_after)
            print(f"OK   {base}  ->  {out_filename}")
        except KeyError as e:
            print(f"SKIP {base}: {e}")
        except Exception as e:
            print(f"FAIL {base}: {e}")


if __name__ == "__main__":
    input_dir  = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\gun_conditioning_comb"
    output_dir = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun_pipeline_until_max"
    process_folder(input_dir, output_dir, n_after=N_AFTER_MAX)
