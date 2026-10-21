"""
MiniRocket + Ridge baseline for the electron-gun voltage-change task.

Goal
----
Provide a fast, deterministic, non-neural baseline to compare against the
CNN / LSTM models. MiniRocket is a *feature extractor* (a fixed convolutional
transform), NOT a classifier, so the pipeline is:

    MiniRocketMultivariate  ->  StandardScaler(with_mean=False)  ->  linear classifier

(this mirrors the official sktime recipe:
 https://www.sktime.net/docs/examples/transformation/minirocket/)

Two classifiers are trained on the SAME (shared) MiniRocket features and
reported side by side (see CLASSIFIERS):
  * ridge        -- sklearn RidgeClassifier (lsqr solver); the sktime default.
  * sgd_logistic -- sklearn SGDClassifier(loss='log_loss'); the fast,
                    sktime-recommended choice for >10k examples.
Both use class_weight='balanced' for the ~5% positive class and are compared
via a grouped bar chart in the aggregate figures folder.

The sktime recipe uses ``RidgeClassifierCV``, whose leave-one-out GCV needs
~3 GB and is impractical at our ~24k training windows (sktime itself recommends
it only for <10k examples). We instead use the official sklearn
``RidgeClassifier`` with the memory-light iterative ``lsqr`` solver, sweeping the
same alpha grid and picking alpha on the validation split by minimum log-loss.

RidgeClassifier is not probabilistic (no ``predict_proba``); it exposes a
signed ``decision_function`` score. We therefore select an operating threshold
on the VALIDATION decision scores (max-MCC, matching the CNN/LSTM training
baseline) and freeze it to the test set. We additionally run a cost-sensitivity
sweep (FP:FN = 1,2,5,10,20) exactly like ``calibrate_checkpoints_isotonic.py``,
but operating on the Ridge decision scores instead of calibrated probabilities.

This file is fully self-contained: the data pipeline functions below are copied
VERBATIM from
"POL CNN or LSTM for GIT upload weights activations dropout correlation
 multichannel missing multiseed.py"
so the train/val/test splits and scaling are byte-for-byte identical. The only
change is that ``prepare_all_data`` returns plain numpy arrays instead of torch
tensors (there is no torch dependency here). Parity is proven separately by
``verify_split_parity.py``.
"""

import os
import glob
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
    log_loss,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# Configuration (copied from the CNN/LSTM __main__ block for exact parity)
# ==========================================================================
TRAIN_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\train poster"
TEST_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\test poster"

FEATURE_COLS = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
SEQUENCE_LENGTH = 30
MISSING_VALUE = 0.0
NOISE_K = 1.0

# Same split-seed grid used by the CNN/LSTM multiseed experiments.
SPLIT_SEEDS = [12, 97, 435, 8765, 88]

# Ridge alpha grid (sktime MiniRocket example uses this range). Alpha is
# selected on the validation set by minimum log-loss (see run_single_seed).
RIDGE_ALPHAS = np.logspace(-3, 3, 10)

# Ridge solver. The sktime recipe uses RidgeClassifierCV, but its leave-one-out
# GCV needs ~3 GB and is impractical for our ~24k training windows (sktime
# itself recommends RidgeClassifierCV only for <10k examples). We use the
# official sklearn RidgeClassifier with the memory-light iterative 'lsqr'
# solver and pick alpha on the validation split instead.
RIDGE_SOLVER = "lsqr"

# SGD logistic regression regularization grid (alpha = reg strength; default
# 1e-4). This is the fast, sktime-recommended companion for MiniRocket on
# >10k examples, and (unlike Ridge) yields calibrated-ish probabilities.
SGD_ALPHAS = np.logspace(-6, -1, 6)


def _make_ridge(alpha, seed):
    return RidgeClassifier(alpha=float(alpha), class_weight="balanced",
                           solver=RIDGE_SOLVER)


def _make_sgd_logistic(alpha, seed):
    return SGDClassifier(loss="log_loss", alpha=float(alpha),
                         class_weight="balanced", random_state=seed,
                         max_iter=1000, tol=1e-3)


# Registry of classifiers trained on the (shared) MiniRocket features. Each
# entry: a name, a factory(alpha, seed), and the alpha grid to sweep. All use
# class_weight='balanced' for the ~5% positive class, decision_function scores,
# and validation-set alpha selection by minimum log-loss.
CLASSIFIERS = {
    "ridge": {"factory": _make_ridge, "alphas": RIDGE_ALPHAS},
    "sgd_logistic": {"factory": _make_sgd_logistic, "alphas": SGD_ALPHAS},
}

# Cost-sensitivity grid: FP cost relative to a fixed FN cost of 1.
COST_FP_RATIOS = (1.0, 2.0, 5.0, 10.0, 20.0)
FN_COST = 1.0

# Output locations (next to this script).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(_SCRIPT_DIR, "baseline_minirocket")
AGG_DIR = os.path.join(_SCRIPT_DIR, "baseline_minirocket_aggregate")

# Metrics aggregated across seeds (pr_auc is threshold-free but reported).
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "pr_auc", "mcc"]


# ==========================================================================
# SECTION A -- Data pipeline (VERBATIM port from the CNN/LSTM script)
# Only change: prepare_all_data returns numpy arrays (no torch).
# ==========================================================================
def robust_noise_sigma_mad_diff(series: pd.Series):
    """
    Robust noise sigma estimate using Medium Absolute Deviation of first differences.
    """
    x = pd.to_numeric(series, errors='coerce').dropna()
    if len(x) < 5:
        return np.nan

    dx = x.diff().dropna()
    mad = np.median(np.abs(dx - np.median(dx)))
    sigma = mad / (0.6745 * np.sqrt(2))
    return sigma


def estimate_folder_noise_thresholds(folder_path, cols, k=1.0):
    """
    Computes per-file MAD-diff sigma for each sensor,
    then returns median sigma across files as the threshold.
    threshold = k * median_sigma
    """
    print(f"--- Estimating Noise Thresholds (MAD baseline) from {folder_path} ---")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    sigma_map = {c: [] for c in cols}

    for f in all_files:
        try:
            df = pd.read_csv(f)
            for c in cols:
                if c in df.columns:
                    s = robust_noise_sigma_mad_diff(df[c])
                    if pd.notna(s):
                        sigma_map[c].append(s)
        except:
            pass

    thresholds = {}
    for c in cols:
        if sigma_map[c]:
            base = float(np.median(sigma_map[c]))
            thresholds[c] = k * base
        else:
            thresholds[c] = 0.0

    print(">>> Noise thresholds (MAD diff):")
    for k_, v_ in thresholds.items():
        print(f"  {k_}: {v_:.6f}")

    return thresholds


def should_skip_quiet_negative(df, idx, noise_thresholds, voltage_limit=None):
    """
    Quiet negative definition (your updated rule):
    - VoltageChange == 0 (checked outside)
    - AND all three sensors are quiet based on delta noise:
        |x[t] - x[t-1]| < sigma_x
    - AND voltage[t] < voltage_limit (per-file max)

    Returns True if sample should be skipped.
    """
    required = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
    for c in required:
        if c not in df.columns:
            return False

    if idx <= 0:
        return False

    # Current timestep values
    cur_val = pd.to_numeric(df['GunCurrent.Avg'].iloc[idx], errors='coerce')
    pre_val = pd.to_numeric(df['peg-BL-cc:pressureM'].iloc[idx], errors='coerce')
    rad_val = pd.to_numeric(df['RadiationTotal'].iloc[idx], errors='coerce')

    # Previous timestep values
    cur_prev = pd.to_numeric(df['GunCurrent.Avg'].iloc[idx-1], errors='coerce')
    pre_prev = pd.to_numeric(df['peg-BL-cc:pressureM'].iloc[idx-1], errors='coerce')
    rad_prev = pd.to_numeric(df['RadiationTotal'].iloc[idx-1], errors='coerce')

    if not all(pd.notna(v) for v in [cur_val, pre_val, rad_val, cur_prev, pre_prev, rad_prev]):
        return False

    low_activity = (
        abs(cur_val - cur_prev) < noise_thresholds.get('GunCurrent.Avg', 0) and
        abs(pre_val - pre_prev) < noise_thresholds.get('peg-BL-cc:pressureM', 0) and
        abs(rad_val - rad_prev) < noise_thresholds.get('RadiationTotal', 0)
    )

    if not low_activity:
        return False

    # Per-file voltage condition
    if voltage_limit is not None and 'glassmanDataXfer:hvPsVoltageMeasM' in df.columns:
        volt_val = pd.to_numeric(df['glassmanDataXfer:hvPsVoltageMeasM'].iloc[idx], errors='coerce')
        if pd.notna(volt_val):
            if not (volt_val < voltage_limit):
                return False

    return True


def read_folder_csvs(folder_path):

    ## Reads the csv files from a folder into a list of dataframes.
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            print(df.shape)
            # Store filename for later debug prints
            df.attrs['source_file'] = os.path.basename(filename)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

        print(len(dfs))
    return dfs


def process_single_df_to_sequences(df, scaler, sequence_length,
                                   noise_thresholds=None,
                                   filter_quiet_negatives=True,
                                   missing_value=0.0,
                                   scale_sensors=True):
    # 0. Pre-processing: Find Time Column and Resample for Gaps
    # We look for case-insensitive 'time' column
    df = df.copy()

    time_col = next((col for col in df.columns if col.lower() == 'time'), None)

    # --- RESTORED YOUR ORIGINAL LOGIC HERE ---
    if time_col:
        is_numeric = False
        try:
            # Try converting to numbers first (e.g. "1.0", "2.0")
            df[time_col] = pd.to_numeric(df[time_col])
            is_numeric = True
        except ValueError:
            pass  # Was not simple numbers, might be datetime

        if not is_numeric:
            try:
                # Try converting to Datetime (e.g. "2024-01-01 12:00:00")
                df[time_col] = pd.to_datetime(df[time_col])
                # Convert Datetime to "Seconds since start" so we can do math
                start_time = df[time_col].min()
                df[time_col] = (df[time_col] - start_time).dt.total_seconds()
            except Exception as e:
                print(f"Warning: Could not convert '{time_col}' to numbers. Skipping resampling. Error: {e}")
                time_col = None  # Disable resampling for this file

    if time_col:
        # Drop duplicates to prevent indexing errors
        df = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col)

        # Calculate dominant time step (frequency)
        if len(df) > 1:
            time_diffs = df[time_col].diff()
            # mode() returns a series, take first element. Rounding handles float jitter.
            freq = time_diffs.mode()[0]

            # Ensure freq is valid and positive
            if pd.notna(freq) and freq > 0:
                # Create a complete time grid from start to end
                min_t, max_t = df[time_col].min(), df[time_col].max()

                # Use numpy arange for float safety
                full_time_index = np.arange(min_t, max_t + freq/1000.0, freq)

                # Rounding to avoid float precision mismatch (e.g. 1.0000001 vs 1.0)
                df[time_col] = df[time_col].round(6)
                full_time_index = np.round(full_time_index, 6)

                # Reindex the dataframe.
                df = df.set_index(time_col).reindex(full_time_index).reset_index()
                df = df.rename(columns={'index': time_col, time_col: time_col})

    # 1. Map Target: -1 -> 0, and enforce a 5-step refractory window after each -1
    if 'VoltageChange' in df.columns:
        # Work on a numeric copy
        vc_raw = pd.to_numeric(df['VoltageChange'], errors='coerce').values.copy()
        n = len(vc_raw)

        # mask of indices that should be forced to 0
        force_zero = np.zeros(n, dtype=bool)

        # locations where the *original* label is -1
        neg_indices = np.where(vc_raw == -1)[0]

        # for each -1, force current index + next 5 to 0
        for idx in neg_indices:
            end = min(n, idx + 6)  # idx, idx+1, ..., idx+5
            force_zero[idx:end] = True

        vc_clean = vc_raw.copy()
        # first map -1 -> 0
        vc_clean[vc_clean == -1] = 0
        # then enforce refractory window -> 0, overriding any 1s
        vc_clean[force_zero] = 0

        df['VoltageChange'] = vc_clean

    # --- NEW: Per-file max voltage for TRAIN filtering ---
    file_max_voltage = None
    if 'glassmanDataXfer:hvPsVoltageMeasM' in df.columns:
        v = pd.to_numeric(df['glassmanDataXfer:hvPsVoltageMeasM'], errors='coerce')
        if v.notna().any():
            file_max_voltage = float(v.max())

    # --- For debug printing ---
    file_label = df.attrs.get('source_file', 'unknown_file')

    # --- Skip counters ---
    total_zero_targets = 0
    skipped_quiet_zero = 0

    # 2. Handle Inputs (NO MASKS, NO VOLTAGE INCLUDED IN FEATURES)
    # Channels: [Current, Pressure, Radiation, Prev_A1] = 4 channels
    feature_cols = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
    input_data = []

    for col in feature_cols:
        # Keep NaN as NaN - will be replaced with 0 after normalization
        val = df[col].astype(float)
        input_data.append(val.values)

    # 3. Add Previous Output (A1) to Input Features
    a1_history = df['VoltageChange'].fillna(0).values
    input_data.append(a1_history)

    # Stack features: [Current, Pressure, Radiation, Prev_A1] = 4 channels
    X_raw = np.stack(input_data, axis=1)

    # 4. Apply Scaler (Only to sensor values - indices 0, 1, 2)
    if scale_sensors:
        if scaler is None:
            raise ValueError("scale_sensors=True requires a fitted scaler.")
        nan_mask = np.isnan(X_raw[:, [0, 1, 2]])
        X_sensor_temp = np.nan_to_num(X_raw[:, [0, 1, 2]], nan=0.0)
        X_raw[:, [0, 1, 2]] = scaler.transform(X_sensor_temp)
        for i in range(3):
            X_raw[nan_mask[:, i], i] = missing_value

    # 5. Create Sequences
    if len(X_raw) <= sequence_length:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1))

    X_seq, y_seq = [], []

    y_raw_target = df['VoltageChange'].values

    for i in range(len(X_raw) - sequence_length):
        idx = i + sequence_length
        target_val = y_raw_target[idx]

        # Only train if we have a VALID target (not a filled gap)
        if np.isnan(target_val):
            continue

        # --- NEW: Skip quiet negatives (TRAIN ONLY) ---
        if filter_quiet_negatives and noise_thresholds and target_val == 0:
            total_zero_targets += 1

            if should_skip_quiet_negative(
                df, idx,
                noise_thresholds=noise_thresholds,
                voltage_limit=file_max_voltage  # per-file max
            ):
                skipped_quiet_zero += 1
                continue

        X_seq.append(X_raw[i: i + sequence_length])
        y_seq.append(target_val)

    if len(X_seq) == 0:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1))

    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)


def _apply_scaler_to_sensor_channels(X, scaler, missing_value=0.0):
    """Apply ``scaler`` to the first three sensor channels of ``X`` while
    preserving NaN positions as ``missing_value`` after scaling.
    """
    if X.size == 0:
        return X
    X = X.copy()
    sensor = X[:, :, [0, 1, 2]].astype(float, copy=True)
    nan_mask = np.isnan(sensor)
    sensor_filled = np.where(nan_mask, 0.0, sensor)
    flat = sensor_filled.reshape(-1, 3)
    flat_scaled = scaler.transform(flat)
    sensor_scaled = flat_scaled.reshape(sensor.shape)
    sensor_scaled[nan_mask] = missing_value
    X[:, :, [0, 1, 2]] = sensor_scaled
    return X


def prepare_all_data(train_folder, test_folder, sequence_length,
                     noise_thresholds=None,
                     filter_quiet_negatives=True,
                     missing_value=0.0,
                     split_seed=None):
    """
    Orchestrates data loading, splitting, and scaling. Identical logic to the
    CNN/LSTM script, but returns numpy arrays instead of torch tensors.

    Order of operations (no train -> val/test scaler leakage):
      1. Read CSV files from the train and test folders.
      2. Build sequences for every training file (raw, unscaled, NaN preserved).
      3. Shuffle-split the training sequences into train/val (85/15).
      4. Fit a NaN-aware StandardScaler ONLY on the train-portion sensors.
      5. Apply that scaler to train/val/test and replace NaN with missing_value.

    Returns (data, pos_ratio, scaler) where data holds numpy arrays:
        X_* -> (Samples, Seq_Len, 4), y_* -> (Samples, 1)
    """
    print(f"Loading Data...")

    # --- READ ALL TRAINING FILES ---
    train_dfs = read_folder_csvs(train_folder)
    test_dfs = read_folder_csvs(test_folder)

    print(f"Loaded {len(train_dfs)} training files")

    # --- BUILD UNSCALED TRAIN SEQUENCES (NaN preserved) ---
    print("Processing Train Sequences (unscaled, NaN preserved)...")
    X_train_list, y_train_list = [], []
    for df in train_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler=None, sequence_length=sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=filter_quiet_negatives,
            missing_value=missing_value,
            scale_sensors=False,
        )
        if len(x_s) > 0:
            X_train_list.append(x_s)
            y_train_list.append(y_s)

    if X_train_list:
        X_train_all = np.concatenate(X_train_list)
        y_train_all = np.concatenate(y_train_list)
    else:
        X_train_all = np.empty((0, sequence_length, 4))
        y_train_all = np.empty((0, 1))

    if split_seed is None:
        split_seed = int(os.environ.get("POLGUN_SPLIT_SEED", "97"))

    # --- SHUFFLED TRAIN/VAL SPLIT (BEFORE SCALING) ---
    print(f"Splitting into train/val with shuffled train_test_split (85/15, random_state={split_seed})...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=0.15,
        shuffle=True,
        random_state=split_seed
    )

    # --- BUILD UNSCALED TEST SEQUENCES (NaN preserved) ---
    print("Processing Test Sequences (unscaled, NaN preserved)...")
    X_test_list, y_test_list = [], []
    test_source_files_list = []
    for df in test_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler=None, sequence_length=sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=False,
            missing_value=missing_value,
            scale_sensors=False,
        )
        if len(x_s) > 0:
            X_test_list.append(x_s)
            y_test_list.append(y_s)
            src_name = df.attrs.get('source_file', 'unknown_file')
            test_source_files_list.extend([src_name] * len(x_s))

    if X_test_list:
        X_test_raw = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        X_test_raw = np.empty((0, sequence_length, 4))
        y_test = np.empty((0, 1))

    test_source_files = np.array(test_source_files_list, dtype=object)

    # --- FIT SCALER ON TRAIN SPLIT ONLY (NaN-aware) ---
    print("Fitting NaN-aware Scaler on TRAIN split sensor values only...")
    train_sensor = X_train_raw[:, :, [0, 1, 2]].reshape(-1, 3) if X_train_raw.size else np.empty((0, 3))
    scaler = StandardScaler()
    if train_sensor.size == 0:
        scaler.mean_ = np.zeros(3)
        scaler.scale_ = np.ones(3)
        scaler.var_ = np.ones(3)
        scaler.n_samples_seen_ = np.zeros(3, dtype=int)
        print("  WARNING: training split is empty; using identity scaler.")
    else:
        scaler.mean_ = np.nanmean(train_sensor, axis=0)
        scaler.scale_ = np.nanstd(train_sensor, axis=0, ddof=0)
        # Guard against degenerate (zero-variance / all-NaN) channels
        scaler.scale_ = np.where(
            (~np.isfinite(scaler.scale_)) | (scaler.scale_ == 0),
            1.0,
            scaler.scale_,
        )
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_samples_seen_ = np.sum(~np.isnan(train_sensor), axis=0)
    scaler.n_features_in_ = 3

    print(f"  Scaler mean (TRAIN-only, ignoring NaN): {scaler.mean_}")
    print(f"  Scaler std  (TRAIN-only, ignoring NaN): {scaler.scale_}")

    # --- APPLY SCALER TO TRAIN/VAL/TEST SEQUENCES ---
    X_train = _apply_scaler_to_sensor_channels(X_train_raw, scaler, missing_value=missing_value)
    X_val = _apply_scaler_to_sensor_channels(X_val_raw, scaler, missing_value=missing_value)
    X_test = _apply_scaler_to_sensor_channels(X_test_raw, scaler, missing_value=missing_value)

    # --- CLASS BALANCE (informational; Ridge uses class_weight='balanced') ---
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_ratio = float(n_neg / n_pos) if n_pos > 0 else 1.0

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'test_source_files': test_source_files,
    }

    # Validation / test class counts
    n_pos_val = np.sum(y_val == 1)
    n_neg_val = np.sum(y_val == 0)
    n_pos_test = np.sum(y_test == 1)
    n_neg_test = np.sum(y_test == 0)

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Training Set:   {data['X_train'].shape}  (Samples, Seq_Len, Channels)")
    print(f"Validation Set: {data['X_val'].shape}")
    print(f"Test Set:       {data['X_test'].shape}")
    print("-" * 50)
    print(f"Shuffled Split: 85% train / 15% val (random_state={split_seed})")
    print("-" * 50)
    print(f"CLASS DISTRIBUTION:")
    print(f"  TRAIN:      Stable(0)={int(n_neg):>6}  |  Increase(1)={int(n_pos):>6}  |  1-ratio={100*n_pos/(n_pos+n_neg):.2f}%")
    print(f"  VALIDATION: Stable(0)={int(n_neg_val):>6}  |  Increase(1)={int(n_pos_val):>6}  |  1-ratio={100*n_pos_val/(n_pos_val+n_neg_val+1e-8):.2f}%")
    print(f"  TEST:       Stable(0)={int(n_neg_test):>6}  |  Increase(1)={int(n_pos_test):>6}  |  1-ratio={100*n_pos_test/(n_pos_test+n_neg_test+1e-8):.2f}%")
    print("-" * 50)
    print(f"neg/pos ratio (Train): {pos_ratio:.4f}")
    print("=" * 50 + "\n")

    return data, pos_ratio, scaler


# ==========================================================================
# SECTION B -- Threshold / metric helpers
# Ported from calibrate_checkpoints_isotonic.py, operating on decision SCORES
# (RidgeClassifier decision_function) instead of calibrated probabilities.
# The logic is identical; only the variable name "probs" -> "scores".
# ==========================================================================
def candidate_thresholds(scores, include_all_negative=True):
    """Sorted unique scores as candidate decision thresholds.

    Using ``preds = scores >= t`` with each unique score as ``t`` sweeps every
    distinct partition from all-positive (t = min) upward. Optionally appends a
    value above the max to also allow the all-negative partition.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return np.array([0.0])
    uniq = np.unique(scores)
    if include_all_negative:
        top = np.nextafter(uniq[-1], np.inf)
        uniq = np.concatenate([uniq, [top]])
    return uniq


def confusion_counts(y_true, preds):
    """Return (tn, fp, fn, tp) as ints, robust to single-class inputs."""
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return tn, fp, fn, tp


def safe_mcc(y_true, preds):
    y_true = np.asarray(y_true, dtype=int).ravel()
    preds = np.asarray(preds, dtype=int).ravel()
    n = len(preds)
    if preds.sum() in (0, n) or y_true.sum() in (0, n):
        return 0.0
    return float(matthews_corrcoef(y_true, preds))


def classification_metrics_at(scores, y_true, threshold):
    """Precision/recall/F1/MCC + confusion counts at a fixed threshold."""
    scores = np.asarray(scores, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, preds)
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "n_samples": int(n),
        "accuracy": float((preds == y_true).mean()) if n else float("nan"),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "mcc": safe_mcc(y_true, preds),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "predicted_positive": int(preds.sum()),
    }


def select_max_mcc_threshold(scores, y_true):
    """Pick the max-MCC threshold from decision scores.

    Deterministic, recall-favoring tie-break: candidates are scanned in
    ascending order and the FIRST (smallest) threshold achieving the maximum
    MCC is chosen (a smaller threshold predicts more positives -> favors
    recall), matching the CNN/LSTM training program's behavior.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    cands = candidate_thresholds(scores, include_all_negative=False)
    best_t = float(cands[0])
    best_mcc = -np.inf
    rows = []
    for t in cands:
        preds = (scores >= t).astype(int)
        mcc = safe_mcc(y_true, preds)
        rows.append((float(t), mcc))
        if mcc > best_mcc:  # strict '>' keeps the smallest t on ties
            best_mcc = mcc
            best_t = float(t)
    return best_t, float(best_mcc), rows


def min_cost_threshold(scores, y_true, c_fp, c_fn=FN_COST):
    """Empirical minimum-cost threshold on the given (validation) scores.

    Cost = c_fn * FN + c_fp * FP. Scans all candidate thresholds (including the
    all-negative partition) and returns the smallest threshold achieving the
    minimum total cost (recall-favoring tie-break).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    cands = candidate_thresholds(scores, include_all_negative=True)
    best_t = float(cands[0])
    best_cost = np.inf
    for t in cands:
        preds = (scores >= t).astype(int)
        _, fp, fn, _ = confusion_counts(y_true, preds)
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:  # strict '<' keeps smallest t on ties
            best_cost = cost
            best_t = float(t)
    return best_t, float(best_cost)


def cost_at_threshold(scores, y_true, threshold, c_fp, c_fn=FN_COST):
    """Evaluate cost + classification metrics at a fixed threshold."""
    m = classification_metrics_at(scores, y_true, threshold)
    total_cost = c_fn * m["fn"] + c_fp * m["fp"]
    m["c_fn"] = float(c_fn)
    m["c_fp"] = float(c_fp)
    m["total_cost"] = float(total_cost)
    m["mean_cost_per_sample"] = float(total_cost / m["n_samples"]) if m["n_samples"] else float("nan")
    return m


# ==========================================================================
# SECTION C -- MiniRocket + Ridge for one split seed
# ==========================================================================
def _import_minirocket():
    """Import MiniRocketMultivariate from whichever sktime path is available."""
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except Exception:
        pass
    try:
        from sktime.transformations.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except Exception as exc:
        raise ImportError(
            "Could not import MiniRocketMultivariate from sktime. "
            "Install with `pip install sktime` (numba is pulled in automatically)."
        ) from exc


def to_channel_major(X):
    """(N, seq_len, channels) -> (N, channels, seq_len) as float32 for sktime."""
    return np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))


def _validation_probabilities(clf, X):
    """Probabilities for log-loss. Prefer predict_proba; else sigmoid(decision)."""
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X), dtype=float)[:, 1]
    # Ridge has no predict_proba; map decision scores to (0, 1) for NLL comparison.
    scores = np.asarray(clf.decision_function(X), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))


def _fit_select_alpha(spec, F_train_s, y_train, F_val_s, y_val, seed):
    """Sweep the alpha grid; select the model by minimum validation log-loss.

    For each alpha the classifier is trained on train and scored on validation
    with mean log-loss (negative log-likelihood). The alpha with the lowest
    validation log-loss is kept. Decision threshold is still chosen later by
    validation max-MCC (see _evaluate_scores). Average precision is reported
    but not used for alpha selection.

    Returns (best_clf, alpha_selected, best_val_logloss, best_val_ap).
    """
    import gc

    best_loss = np.inf
    best_clf = None
    alpha_selected = float("nan")
    best_ap = float("nan")
    # Keep arrays float32 to limit peak memory during weighted ridge fits.
    F_train_s = np.asarray(F_train_s, dtype=np.float32)
    F_val_s = np.asarray(F_val_s, dtype=np.float32)
    for a in spec["alphas"]:
        cand = spec["factory"](a, seed)
        cand.fit(F_train_s, y_train)
        s_val_cand = np.asarray(cand.decision_function(F_val_s), dtype=float).ravel()
        if len(np.unique(y_val)) > 1:
            probs = _validation_probabilities(cand, F_val_s)
            # clip for numerical safety inside log_loss
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            val_loss = float(log_loss(y_val, probs, labels=[0, 1]))
            ap = float(average_precision_score(y_val, s_val_cand))
        else:
            val_loss = float("nan")
            ap = float("nan")
        print(f"    alpha={a:<10.4g} val_logloss={val_loss:.4f}  val_AP={ap:.4f}")
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_clf = cand
            alpha_selected = float(a)
            best_ap = ap
        elif cand is not best_clf:
            del cand
        gc.collect()
    if best_clf is None:  # degenerate val (single class): fall back to mid alpha
        alpha_selected = float(spec["alphas"][len(spec["alphas"]) // 2])
        best_clf = spec["factory"](alpha_selected, seed).fit(F_train_s, y_train)
        best_loss = float("nan")
        best_ap = float("nan")
    return best_clf, alpha_selected, float(best_loss), float(best_ap)


def _evaluate_scores(split_seed, s_val, y_val, s_test, y_test,
                     alpha_selected, best_ap, pos_ratio):
    """Threshold selection + metrics + cost sweep from decision scores."""
    # --- Max-MCC threshold on validation scores, frozen to test ---
    mcc_t, mcc_val_best, mcc_rows = select_max_mcc_threshold(s_val, y_val)
    val_metrics = classification_metrics_at(s_val, y_val, mcc_t)
    test_metrics = classification_metrics_at(s_test, y_test, mcc_t)

    # Threshold-free PR-AUC (average precision) on the raw scores.
    val_metrics["pr_auc"] = (float(average_precision_score(y_val, s_val))
                             if len(np.unique(y_val)) > 1 else float("nan"))
    test_metrics["pr_auc"] = (float(average_precision_score(y_test, s_test))
                              if len(np.unique(y_test)) > 1 else float("nan"))

    # --- Cost-sensitivity sweep (FN cost fixed at 1; FP cost swept) ---
    cost_rows = []
    cost_selected_thresholds = {}
    for c_fp in COST_FP_RATIOS:
        emp_t, _ = min_cost_threshold(s_val, y_val, c_fp, FN_COST)
        cost_selected_thresholds[c_fp] = emp_t
        for split, scores, y in (("val", s_val, y_val), ("test", s_test, y_test)):
            row = cost_at_threshold(scores, y, emp_t, c_fp, FN_COST)
            row.update({
                "split_seed": int(split_seed),
                "fp_fn_ratio": f"{int(c_fp)}:1",
                "c_fp": float(c_fp),
                "c_fn": float(FN_COST),
                "split": split,
            })
            cost_rows.append(row)

    return {
        "split_seed": int(split_seed),
        "alpha_selected": alpha_selected,
        "val_ap_selected": float(best_ap),
        "pos_ratio": pos_ratio,
        "chosen_threshold": float(mcc_t),
        "val_best_mcc": float(mcc_val_best),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cost_rows": cost_rows,
        "cost_selected_thresholds": cost_selected_thresholds,
        # arrays kept for plotting
        "_s_val": s_val, "_y_val": y_val,
        "_s_test": s_test, "_y_test": y_test,
        "_mcc_rows": mcc_rows,
    }


def run_single_seed(split_seed, noise_thresholds, MiniRocketMultivariate):
    """MiniRocket features (computed once) -> every classifier in CLASSIFIERS.

    Returns {classifier_name: result_dict} for one split seed.
    """
    print("\n" + "#" * 70)
    print(f"# MiniRocket baseline -- split_seed={split_seed}")
    print("#" * 70)

    data, pos_ratio, _scaler = prepare_all_data(
        TRAIN_DIR, TEST_DIR, SEQUENCE_LENGTH,
        noise_thresholds=noise_thresholds,
        filter_quiet_negatives=True,
        missing_value=MISSING_VALUE,
        split_seed=split_seed,
    )

    # --- Convert to sktime channel-major layout [N, channels, seq_len] ---
    X_train = to_channel_major(data['X_train'])
    X_val = to_channel_major(data['X_val'])
    X_test = to_channel_major(data['X_test'])
    y_train = data['y_train'].ravel().astype(int)
    y_val = data['y_val'].ravel().astype(int)
    y_test = data['y_test'].ravel().astype(int)

    # --- MiniRocket transform (fit on TRAIN only; unsupervised) computed ONCE
    #     and shared by all classifiers to avoid repeating the expensive step. ---
    print("Fitting MiniRocketMultivariate on train (Numba compiles on first import)...")
    mr = MiniRocketMultivariate(random_state=split_seed)
    mr.fit(X_train)

    feat_scaler = StandardScaler(with_mean=False)
    F_train_s = feat_scaler.fit_transform(
        np.asarray(mr.transform(X_train), dtype=np.float32)).astype(np.float32)
    F_val_s = feat_scaler.transform(
        np.asarray(mr.transform(X_val), dtype=np.float32)).astype(np.float32)
    F_test_s = feat_scaler.transform(
        np.asarray(mr.transform(X_test), dtype=np.float32)).astype(np.float32)
    print(f"  MiniRocket features: train={F_train_s.shape}, val={F_val_s.shape}, test={F_test_s.shape}")

    results = {}
    for clf_name, spec in CLASSIFIERS.items():
        print(f"\n  --- Classifier: {clf_name} (select alpha on validation log-loss) ---")
        clf, alpha_selected, best_val_logloss, best_ap = _fit_select_alpha(
            spec, F_train_s, y_train, F_val_s, y_val, split_seed)
        print(f"  [{clf_name}] selected alpha = {alpha_selected:.6g} "
              f"(val log-loss = {best_val_logloss:.4f}, val AP = {best_ap:.4f})")

        s_val = np.asarray(clf.decision_function(F_val_s), dtype=float).ravel()
        s_test = np.asarray(clf.decision_function(F_test_s), dtype=float).ravel()
        results[clf_name] = _evaluate_scores(
            split_seed, s_val, y_val, s_test, y_test,
            alpha_selected, best_ap, pos_ratio)

    del F_train_s, F_val_s, F_test_s
    return results


# ==========================================================================
# SECTION D -- Plotting (standalone MiniRocket figures)
# ==========================================================================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_run_figures(result, fig_dir, clf_label=""):
    _ensure_dir(fig_dir)
    s_val, y_val = result["_s_val"], result["_y_val"]
    s_test, y_test = result["_s_test"], result["_y_test"]
    thr = result["chosen_threshold"]
    tag = f"{clf_label} seed {result['split_seed']}".strip()

    # 1. Validation PR curve with chosen max-MCC threshold marked.
    if len(np.unique(y_val)) > 1:
        precision, recall, pr_thresholds = precision_recall_curve(y_val, s_val)
        ap = average_precision_score(y_val, s_val)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2, label=f"PR curve (AP={ap:.3f})")
        # locate operating point at chosen threshold
        vm = result["val_metrics"]
        ax.scatter([vm["recall"]], [vm["precision"]], color="red", zorder=5,
                   label=f"max-MCC t={thr:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Validation PR curve ({tag})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "val_pr_curve.png"), dpi=150)
        plt.close(fig)

    # 2. MCC vs threshold on validation.
    if result["_mcc_rows"]:
        rows = np.array(result["_mcc_rows"], dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rows[:, 0], rows[:, 1], lw=1.5)
        ax.axvline(thr, color="red", ls="--", label=f"chosen t={thr:.3f}")
        ax.set_xlabel("Decision-score threshold")
        ax.set_ylabel("MCC")
        ax.set_title(f"Validation MCC vs threshold ({tag})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "val_mcc_vs_threshold.png"), dpi=150)
        plt.close(fig)

    # 3. Test confusion matrix at the frozen threshold.
    tm = result["test_metrics"]
    cm = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Test confusion ({tag}, t={thr:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "test_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 4. Decision-score histogram by true class (test set).
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = s_test[y_test == 1]
    neg = s_test[y_test == 0]
    bins = np.linspace(min(s_test.min(), -1e-9), max(s_test.max(), 1e-9), 40)
    ax.hist(neg, bins=bins, alpha=0.6, label="True 0 (stable)", color="tab:blue")
    ax.hist(pos, bins=bins, alpha=0.6, label="True 1 (increase)", color="tab:orange")
    ax.axvline(thr, color="red", ls="--", label=f"threshold t={thr:.3f}")
    ax.set_xlabel("Decision score")
    ax.set_ylabel("Count")
    ax.set_title(f"Test decision scores by class ({tag})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "score_hist_by_class.png"), dpi=150)
    plt.close(fig)

    # 5. Cost-vs-threshold curves (val + test) with per-ratio min-cost markers.
    for split, scores, y in (("val", s_val, y_val), ("test", s_test, y_test)):
        cands = candidate_thresholds(scores, include_all_negative=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for c_fp in COST_FP_RATIOS:
            costs = []
            for t in cands:
                preds = (scores >= t).astype(int)
                _, fp, fn, _ = confusion_counts(y, preds)
                costs.append(FN_COST * fn + c_fp * fp)
            ax.plot(cands, costs, lw=1.2, label=f"FP:FN={int(c_fp)}:1")
            # mark the VALIDATION-selected (frozen) threshold on this split
            sel_t = result["cost_selected_thresholds"][c_fp]
            preds = (scores >= sel_t).astype(int)
            _, fp, fn, _ = confusion_counts(y, preds)
            ax.scatter([sel_t], [FN_COST * fn + c_fp * fp], s=25, zorder=5)
        ax.set_xlabel("Decision-score threshold")
        ax.set_ylabel("Total cost")
        ax.set_title(f"Cost vs threshold ({split}, {tag})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"cost_curve_{split}.png"), dpi=150)
        plt.close(fig)


def plot_aggregate_figures(per_run_metrics, cost_summary, fig_dir, clf_label="MiniRocket"):
    _ensure_dir(fig_dir)

    # Headline bar chart: mean +/- SD of test metrics across seeds.
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    means = [np.mean([r[m] for r in per_run_metrics]) for m in metrics]
    sds = [np.std([r[m] for r in per_run_metrics], ddof=0) for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=sds, capsize=5, color="tab:green", alpha=0.8)
    for xi, m, s in zip(x, means, sds):
        ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{clf_label} test metrics (mean +/- SD, n={len(per_run_metrics)} seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "test_metrics_bar_mean_sd.png"), dpi=150)
    plt.close(fig)

    # Cost sensitivity: mean +/- SD test total_cost vs FP:FN ratio.
    ratios = [f"{int(c)}:1" for c in COST_FP_RATIOS]
    means_c, sds_c = [], []
    for r in ratios:
        vals = [row["total_cost"] for row in cost_summary
                if row["split"] == "test" and row["fp_fn_ratio"] == r]
        means_c.append(np.mean(vals) if vals else np.nan)
        sds_c.append(np.std(vals, ddof=0) if vals else np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(ratios))
    ax.errorbar(x, means_c, yerr=sds_c, marker="o", capsize=5, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_xlabel("FP:FN cost ratio")
    ax.set_ylabel("Test total cost")
    ax.set_title(f"{clf_label} cost sensitivity (mean +/- SD across seeds)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "cost_sensitivity_mean_sd.png"), dpi=150)
    plt.close(fig)


# ==========================================================================
# SECTION E -- Aggregation helpers
# ==========================================================================
def _summarize(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan, "median": np.nan, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _aggregate_classifier(clf_name, per_run_test, per_run_val, cost_rows):
    """Write per-classifier aggregate CSV/JSON + figures. Returns summary dict."""
    agg_dir = _ensure_dir(os.path.join(AGG_DIR, clf_name))

    per_run_test_df = pd.DataFrame(per_run_test)
    per_run_test_df.to_csv(os.path.join(agg_dir, "all_runs_test_metrics.csv"), index=False)
    pd.DataFrame(per_run_val).to_csv(
        os.path.join(agg_dir, "all_runs_validation_metrics.csv"), index=False)
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(os.path.join(agg_dir, "all_runs_cost_sensitivity.csv"), index=False)

    # Summary of test metrics (mean/sd/min/max/median across seeds).
    summary = {m: _summarize(per_run_test_df[m].values) for m in METRIC_NAMES}
    summary["chosen_threshold"] = _summarize(per_run_test_df["chosen_threshold"].values)
    write_json(os.path.join(agg_dir, "aggregate_test_metrics_mean_sd.json"), summary)

    summary_rows = []
    for m, stats in summary.items():
        row = {"metric": m}
        row.update(stats)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(agg_dir, "aggregate_test_metrics_mean_sd.csv"), index=False)

    # Cost-sensitivity summary (mean/sd across seeds, per split + ratio).
    cost_summary_rows = []
    for split in ("val", "test"):
        for c_fp in COST_FP_RATIOS:
            ratio = f"{int(c_fp)}:1"
            subset = cost_df[(cost_df["split"] == split) & (cost_df["fp_fn_ratio"] == ratio)]
            for field in ("total_cost", "mean_cost_per_sample", "precision", "recall", "f1", "mcc", "accuracy"):
                stats = _summarize(subset[field].values)
                cost_summary_rows.append({
                    "split": split, "fp_fn_ratio": ratio, "field": field, **stats,
                })
    pd.DataFrame(cost_summary_rows).to_csv(
        os.path.join(agg_dir, "summary_cost_sensitivity_mean_sd.csv"), index=False)

    plot_aggregate_figures(per_run_test, cost_rows, os.path.join(agg_dir, "figures"),
                           clf_label=f"MiniRocket+{clf_name}")
    return summary


def plot_classifier_comparison(summaries, fig_dir):
    """Grouped bar chart comparing all classifiers on test metrics."""
    _ensure_dir(fig_dir)
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    names = list(summaries.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        means = [summaries[name][m]["mean"] for m in metrics]
        sds = [summaries[name][m]["sd"] for m in metrics]
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=sds, capsize=4, label=name, alpha=0.85)
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mv + 0.01, f"{mv:.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"MiniRocket baselines: test metrics (mean +/- SD, n={len(SPLIT_SEEDS)} seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "classifier_comparison_test_metrics.png"), dpi=150)
    plt.close(fig)


# ==========================================================================
# SECTION F -- Main
# ==========================================================================
def main():
    _ensure_dir(RUNS_DIR)
    _ensure_dir(AGG_DIR)

    MiniRocketMultivariate = _import_minirocket()

    # Noise thresholds (identical to CNN/LSTM: TRAIN folder, MAD, k=1.0).
    noise_thresholds = estimate_folder_noise_thresholds(TRAIN_DIR, FEATURE_COLS, k=NOISE_K)

    # Per-classifier accumulators across seeds.
    acc = {name: {"test": [], "val": [], "cost": []} for name in CLASSIFIERS}

    for seed in SPLIT_SEEDS:
        seed_results = run_single_seed(seed, noise_thresholds, MiniRocketMultivariate)

        for clf_name, result in seed_results.items():
            run_dir = _ensure_dir(os.path.join(RUNS_DIR, clf_name, f"run_seed{seed}"))
            fig_dir = os.path.join(run_dir, "figures")

            val_sel = {
                "classifier": clf_name,
                "criterion": "max_mcc_on_decision_scores",
                "split_seed": seed,
                "chosen_threshold": result["chosen_threshold"],
                "val_best_mcc": result["val_best_mcc"],
                "alpha_selected": result["alpha_selected"],
                "alpha_selection_metric": "val_log_loss",
                "val_ap_selected": result["val_ap_selected"],
                "validation": result["val_metrics"],
            }
            write_json(os.path.join(run_dir, "val_threshold_selection.json"), val_sel)

            test_out = dict(result["test_metrics"])
            test_out["classifier"] = clf_name
            test_out["split_seed"] = seed
            test_out["chosen_threshold"] = result["chosen_threshold"]
            write_json(os.path.join(run_dir, "test_metrics.json"), test_out)

            pd.DataFrame(result["cost_rows"]).to_csv(
                os.path.join(run_dir, "cost_sensitivity.csv"), index=False)

            plot_run_figures(result, fig_dir, clf_label=clf_name)

            tm = {"split_seed": seed}
            tm.update({m: result["test_metrics"].get(m, np.nan) for m in METRIC_NAMES})
            tm["chosen_threshold"] = result["chosen_threshold"]
            tm["alpha_selected"] = result["alpha_selected"]
            acc[clf_name]["test"].append(tm)

            vm = {"split_seed": seed}
            vm.update({m: result["val_metrics"].get(m, np.nan) for m in METRIC_NAMES})
            vm["chosen_threshold"] = result["chosen_threshold"]
            acc[clf_name]["val"].append(vm)

            acc[clf_name]["cost"].extend(result["cost_rows"])

            print(f"[seed {seed}][{clf_name}] TEST: acc={tm['accuracy']:.4f} "
                  f"prec={tm['precision']:.4f} rec={tm['recall']:.4f} "
                  f"f1={tm['f1']:.4f} mcc={tm['mcc']:.4f} (t={result['chosen_threshold']:.4f})")

    # ---------------- Per-classifier aggregation ----------------
    summaries = {}
    for clf_name in CLASSIFIERS:
        summaries[clf_name] = _aggregate_classifier(
            clf_name, acc[clf_name]["test"], acc[clf_name]["val"], acc[clf_name]["cost"])

    # ---------------- Cross-classifier comparison figure ----------------
    plot_classifier_comparison(summaries, os.path.join(AGG_DIR, "figures"))

    # ---------------- Console summary ----------------
    for clf_name in CLASSIFIERS:
        print("\n" + "=" * 60)
        print(f"AGGREGATE TEST METRICS -- {clf_name} (mean +/- SD across seeds)")
        print("=" * 60)
        for m in METRIC_NAMES:
            s = summaries[clf_name][m]
            print(f"  {m:>10}: {s['mean']:.4f} +/- {s['sd']:.4f}  "
                  f"(min={s['min']:.4f}, max={s['max']:.4f})")
    print("=" * 60)
    print(f"Per-run outputs : {RUNS_DIR}\\<classifier>\\run_seed<seed>")
    print(f"Aggregate output: {AGG_DIR}\\<classifier>")
    print(f"Comparison figure: {os.path.join(AGG_DIR, 'figures', 'classifier_comparison_test_metrics.png')}")


if __name__ == "__main__":
    main()
