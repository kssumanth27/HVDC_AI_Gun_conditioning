"""
polgun_train.py
===============

Training script for the HVDC polarized-gun conditioning models (1D-CNN or LSTM).

Each run:
  1. Reads the training CSVs, builds 4-channel sliding windows
     [Current, Pressure, Radiation, Prev_A1], and splits them 85/15 into
     train/validation (split seed).
  2. Fits a NaN-aware StandardScaler on the training split only.
  3. Trains with BCEWithLogitsLoss(pos_weight = N0/N1) and Adam, with early
     stopping on validation loss (best-epoch weights restored).
  4. Sweeps the decision threshold on validation and commits t* (max MCC).
  5. Saves reports, figures, and the checkpoint (weights, scaler,
     hyperparameters, history, t*).

Running this file with no environment variables launches the full
5 weight-init seeds x 5 split seeds grid (25 isolated subprocesses) and then
writes the validation mean/SD across runs. Test-set evaluation is done
separately by ``polgun_test.py``.

Usage
-----
    python polgun_train.py
"""

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
os.environ['KMP_WARNINGS'] = '0'
os.environ['MKL_CBWR'] = 'AUTO'
os.environ['MKL_WARN'] = '0'

import copy
import glob
import json
import platform
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None
try:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass


# ==========================================
# CONFIGURATION
# ==========================================

TRAIN_DIR = (
    r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2"
    r"\polgun v8 until max conditioning\v8 spikes cleaned until max\train poster"
)
# Not read during training. Stored in the checkpoint as the default test
# folder offered by polgun_test.py.
TEST_DIR = (
    r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2"
    r"\polgun v8 until max conditioning\v8 spikes cleaned until max\test poster"
)

MODEL_TYPE = 'cnn'  # 'cnn' or 'lstm'

HYPERPARAMS = {
    'sequence_length': 30,
    'learning_rate': 0.000005,
    'epochs': 50,
    'batch_size': 128,
    'n_filters1': 128,
    'n_filters2': 256,
    'kernel_size': 5,
    'pool_size': 2,
    'hidden_dim': 64,  # For LSTM
    'early_stopping_patience': 15,
    'missing_value': 0,  # Value assigned to missing sensor readings after normalization
    # Decision threshold selected on validation; choices: max_mcc, max_f1, fixed_0_5
    'threshold_criterion': 'max_mcc',
}

WEIGHT_INIT_SEEDS = [33, 545, 97, 123, 999]
SPLIT_SEEDS = [12, 97, 435, 8765, 88]

CHANNEL_NAMES = [
    "Current",
    "Pressure",
    "Radiation",
    "Prev_A1",
]

FEATURE_COLS = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]


# ==========================================
# 0. Seeds and Output Handling
# ==========================================

ACTIVE_OUTPUT_FOLDER = None
_AUTO_FIGURE_COUNTER = 0


def set_all_seeds(seed):
    """Set all RNG seeds used by this script for reproducible model initialization/training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_active_output_folder(output_folder):
    """Route any unsaved plt.show() figures to the active run folder in batch mode."""
    global ACTIVE_OUTPUT_FOLDER
    ACTIVE_OUTPUT_FOLDER = output_folder


def _save_unsaved_figures_instead_of_show(*args, **kwargs):
    """Save unsaved figures instead of opening GUI windows."""
    global _AUTO_FIGURE_COUNTER
    if ACTIVE_OUTPUT_FOLDER:
        figures_dir = os.path.join(ACTIVE_OUTPUT_FOLDER, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if getattr(fig, "_saved_by_script", False):
                continue
            _AUTO_FIGURE_COUNTER += 1
            filepath = os.path.join(figures_dir, f"auto_saved_plot_{_AUTO_FIGURE_COUNTER:04d}.png")
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            fig._saved_by_script = True
            print(f"Saved: {filepath}")
    plt.close('all')


plt.show = _save_unsaved_figures_instead_of_show


def create_output_folder(prefix="polgun_cnn"):
    """
    Creates a timestamped output folder with a figures subfolder.

    Args:
        prefix: Custom prefix for the folder name (e.g., 'polgun_cnn', 'polgun_lstm')

    Returns:
        folder_name: Path to the created output folder
    """
    forced_folder = os.environ.get("POLGUN_OUTPUT_FOLDER")
    folder_name = forced_folder if forced_folder else f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    figures_path = os.path.join(folder_name, "figures")
    os.makedirs(figures_path, exist_ok=True)
    set_active_output_folder(folder_name)
    print(f">>> Created output folder: {folder_name}")
    return folder_name


def save_figure(fig, output_folder, filename):
    """
    Saves a matplotlib figure to the figures subfolder as PNG.

    Args:
        fig: matplotlib figure object
        output_folder: path to the output folder
        filename: name for the file (without extension)
    """
    filepath = os.path.join(output_folder, "figures", f"{filename}.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    fig._saved_by_script = True
    print(f"Saved: {filepath}")


def save_text_report(output_folder, filename, content):
    """Save a string to a .txt file inside the output folder."""
    filepath = os.path.join(output_folder, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filepath}")


# ==========================================
# 1. Reports
# ==========================================

def save_epoch_history_report(output_folder, history):
    """Write per-epoch training metrics to a JSON file."""
    n_epochs = len(history['train_loss'])
    records = []
    for i in range(n_epochs):
        rec = {
            'epoch': i + 1,
            'train_loss': history['train_loss'][i],
            'val_loss': history['val_loss'][i],
            'train_acc': history.get('train_acc', [None]*n_epochs)[i],
            'val_acc': history['val_acc'][i],
        }
        records.append(rec)
    filepath = os.path.join(output_folder, "epoch_history.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {filepath}")


def compute_validation_pr_threshold(
    models,
    forward_fn,
    data,
    output_folder=None,
    criterion: str = "max_mcc",
    default_threshold: float = 0.5,
):
    """
    Run the trained model on validation, build the precision-recall curve, and
    score every candidate threshold with F1 and MCC. Pick a single decision
    threshold t* using the chosen criterion (max_mcc, max_f1, or fixed).

    Saves three artifacts in ``output_folder``:
      - ``val_pr_curve_points.csv``: row per candidate threshold with columns
        ``threshold, precision, recall, f1, mcc, predicted_positive_count``.
        Convention: row ``i`` uses sklearn's ``thresholds[i]`` paired with
        ``precision[i]`` and ``recall[i]``; the trivial ``(P=1, R=0)`` tail
        point is dropped because it has no associated threshold.
      - ``figures/val_mcc_vs_threshold.png``: validation MCC versus threshold,
        with a line at the committed t*.
      - ``val_threshold_selection.json``: the committed ``chosen_threshold``
        (with criterion) and the rows used to choose it.

    Returns ``(chosen_threshold, selection_info)``.
    """
    X_val, y_val = data['X_val'], data['y_val']

    if len(X_val) == 0:
        print("Validation set is empty; cannot compute PR curve.")
        return float(default_threshold), {}

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_val)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    y_true = y_val.cpu().numpy().flatten().astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        print(f"Validation set has only one class (pos={n_pos}, neg={n_neg}); "
              f"PR curve / threshold tuning is undefined. Falling back to {default_threshold}.")
        return float(default_threshold), {}

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    average_precision = float(average_precision_score(y_true, probs))

    n_t = len(thresholds)
    f1_arr = np.zeros(n_t, dtype=float)
    mcc_arr = np.zeros(n_t, dtype=float)
    pred_pos_count = np.zeros(n_t, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(thresholds):
            y_pred = (probs >= t).astype(int)
            pred_pos_count[i] = int(y_pred.sum())
            f1_arr[i] = f1_score(y_true, y_pred, zero_division=0)
            mcc_arr[i] = matthews_corrcoef(y_true, y_pred) if y_pred.sum() not in (0, len(y_pred)) else 0.0

    pr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:n_t],
        "recall": recall[:n_t],
        "f1": f1_arr,
        "mcc": mcc_arr,
        "predicted_positive_count": pred_pos_count,
    })

    if output_folder is not None:
        csv_path = os.path.join(output_folder, "val_pr_curve_points.csv")
        pr_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    idx_max_f1 = int(np.argmax(f1_arr))
    idx_max_mcc = int(np.argmax(mcc_arr))

    def _row_to_dict(idx):
        return {
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_arr[idx]),
            "mcc": float(mcc_arr[idx]),
            "predicted_positive_count": int(pred_pos_count[idx]),
        }

    row_max_f1 = _row_to_dict(idx_max_f1)
    row_max_mcc = _row_to_dict(idx_max_mcc)

    y_pred_05 = (probs >= default_threshold).astype(int)
    metrics_at_default = {
        "threshold": float(default_threshold),
        "precision": float(precision_score(y_true, y_pred_05, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_05, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_05, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred_05)) if y_pred_05.sum() not in (0, len(y_pred_05)) else 0.0,
        "predicted_positive_count": int(y_pred_05.sum()),
    }

    crit = criterion.lower().strip()
    if crit == "max_f1":
        chosen = row_max_f1
    elif crit == "max_mcc":
        chosen = row_max_mcc
    elif crit == "fixed_0_5":
        chosen = metrics_at_default
    else:
        print(f"Unknown threshold criterion '{criterion}'; falling back to max_mcc.")
        crit = "max_mcc"
        chosen = row_max_mcc
    chosen_threshold = float(chosen["threshold"])

    selection_info = {
        "criterion": crit,
        "chosen_threshold": chosen_threshold,
        "chosen_row": chosen,
        "argmax_f1": row_max_f1,
        "argmax_mcc": row_max_mcc,
        "metrics_at_0_5": metrics_at_default,
        "average_precision": average_precision,
        "n_validation_samples": int(len(probs)),
        "validation_class_counts": {"stable_0": n_neg, "increase_1": n_pos},
        "csv_path": os.path.join(output_folder, "val_pr_curve_points.csv") if output_folder else None,
        "note": (
            "Row i in val_pr_curve_points.csv uses sklearn thresholds[i] paired with "
            "precision[i] and recall[i]; the trivial (P=1, R=0) tail point with no threshold is dropped."
        ),
    }

    if output_folder is not None:
        json_path = os.path.join(output_folder, "val_threshold_selection.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(selection_info, f, indent=2)
        print(f"Saved: {json_path}")

        fig, ax_t = plt.subplots(figsize=(6, 4))
        ax_t.plot(thresholds, mcc_arr, color="crimson", linewidth=2.0, label="Validation MCC")
        ax_t.axvline(chosen_threshold, color="black", linestyle="--", linewidth=1.2,
                     label=f"t* = {chosen_threshold:.3f}")
        ax_t.set_xlabel("Threshold")
        ax_t.set_ylabel("MCC")
        ax_t.set_xlim(0.0, 1.0)
        ax_t.set_ylim(min(-0.05, float(np.min(mcc_arr)) - 0.05), 1.02)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(loc="best")
        plt.tight_layout()
        save_figure(fig, output_folder, "val_mcc_vs_threshold")
        plt.close(fig)

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD SUMMARY")
    print("=" * 60)
    print(f"  Committed criterion: {crit}  ->  t* = {chosen_threshold:.6f}")
    print(f"  P={chosen['precision']:.4f}  R={chosen['recall']:.4f}  "
          f"F1={chosen['f1']:.4f}  MCC={chosen['mcc']:.4f}")
    print("=" * 60 + "\n")

    return chosen_threshold, selection_info


def save_model_architecture_report(output_folder, models, hyperparams, model_type, data):
    """
    Write a structured model architecture / parameter table to JSON,
    suitable for placing beside an architecture diagram in a presentation.
    """
    seq_len = data['X_train'].shape[1]
    input_channels = data['X_train'].shape[2]

    layers_table = []
    total_params = 0
    for layer_name, layer in models.items():
        layer_trainable = 0
        param_shapes = {}
        for pname, p in layer.named_parameters():
            if p.requires_grad:
                num = p.numel()
                layer_trainable += num
                param_shapes[pname] = list(p.shape)
        total_params += layer_trainable
        layers_table.append({
            'layer_name': layer_name,
            'type': layer.__class__.__name__,
            'param_shapes': param_shapes,
            'trainable_params': layer_trainable,
        })

    report = {
        'model_type': model_type,
        'input_channels': input_channels,
        'sequence_length': seq_len,
        'layers': layers_table,
        'total_trainable_params': total_params,
        'hyperparameters': {
            'learning_rate': hyperparams.get('learning_rate'),
            'epochs': hyperparams.get('epochs'),
            'batch_size': hyperparams.get('batch_size'),
            'n_filters1': hyperparams.get('n_filters1'),
            'n_filters2': hyperparams.get('n_filters2'),
            'kernel_size': hyperparams.get('kernel_size'),
            'pool_size': hyperparams.get('pool_size'),
            'hidden_dim': hyperparams.get('hidden_dim'),
            'early_stopping_patience': hyperparams.get('early_stopping_patience'),
        },
    }
    filepath = os.path.join(output_folder, "model_architecture.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {filepath}")


def get_system_specs():
    """
    Collects hardware and software specs for reproducibility reporting.
    Uses psutil/cpuinfo if installed, falls back gracefully otherwise.
    """
    specs = {
        'os': platform.platform(),
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': None,
        'cpu_model': platform.processor() or 'unknown',
        'cpu_cores_logical': os.cpu_count(),
        'cpu_cores_physical': None,
        'ram_total_gb': None,
    }

    try:
        import sklearn
        specs['sklearn_version'] = sklearn.__version__
    except Exception:
        pass

    try:
        import psutil
        specs['cpu_cores_physical'] = psutil.cpu_count(logical=False)
        specs['ram_total_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        specs['cpu_model'] = info.get('brand_raw', specs['cpu_model'])
        if specs['cpu_cores_physical'] is None:
            specs['cpu_cores_physical'] = info.get('count', None)
    except ImportError:
        pass

    return specs


def save_system_and_timing(output_folder, specs, training_seconds, run_timestamp):
    """
    Saves system specs + training wall-clock time as both JSON and TXT.
    """
    mins, secs = divmod(int(training_seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        formatted = f"{hrs}h {mins}m {secs}s"
    else:
        formatted = f"{mins}m {secs}s"

    report = {
        'system': specs,
        'training_wall_clock_seconds': round(training_seconds, 2),
        'training_wall_clock_formatted': formatted,
        'run_timestamp': run_timestamp,
    }

    json_path = os.path.join(output_folder, "system_and_timing.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {json_path}")

    lines = []
    lines.append("=" * 60)
    lines.append("SYSTEM & TRAINING ENVIRONMENT")
    lines.append("=" * 60)
    lines.append(f"  CPU:              {specs['cpu_model']}")
    cores_str = str(specs.get('cpu_cores_physical') or '?')
    lines.append(f"  Cores:            {cores_str} physical / {specs['cpu_cores_logical']} logical")
    ram_str = f"{specs['ram_total_gb']} GB" if specs['ram_total_gb'] else "unavailable"
    lines.append(f"  RAM:              {ram_str}")
    lines.append(f"  OS:               {specs['os']}")
    lines.append(f"  Python:           {specs['python_version']}")
    lines.append(f"  PyTorch:          {specs['pytorch_version']}")
    lines.append(f"  NumPy:            {specs['numpy_version']}")
    lines.append(f"  pandas:           {specs['pandas_version']}")
    lines.append(f"  scikit-learn:     {specs.get('sklearn_version', 'N/A')}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("TRAINING TIME")
    lines.append("=" * 60)
    lines.append(f"  Wall-clock time:  {formatted} ({training_seconds:.2f} seconds)")
    lines.append(f"  Run started:      {run_timestamp}")
    lines.append("=" * 60)

    txt_path = os.path.join(output_folder, "system_and_timing.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")

    return report


# ==========================================
# 2. Physics Constraints (Helper Functions)
# ==========================================

def calculate_average_step_size(folder_path, voltage_col='glassmanDataXfer:hvPsVoltageMeasM', target_col='VoltageChange'):
    """
    WHAT: Calculates the average step size of the training files for increasing the voltage.
    HOW:  It aligns row t (Command=1) with row t+1 (Voltage).
          It calculates V(t+1) - V(t) for every successful ramp and averages them.

    Inputs: Training folder path, voltage column name, VoltageChange column name.
    Outputs: Average step size (float).
    """
    print(f"--- Calculating Average Step Size from {folder_path} ---")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    all_increases = []

    for filename in all_files:
        try:
            df = pd.read_csv(filename, usecols=[voltage_col, target_col])
            df[voltage_col] = pd.to_numeric(df[voltage_col], errors='coerce')

            # Effect of the command at time t is seen at t+1.
            df['next_voltage'] = df[voltage_col].shift(-1)
            df['forward_step'] = df['next_voltage'] - df[voltage_col]

            # Target=1 and the voltage actually increased (filters out failures/noise).
            valid_increases = df[(df[target_col] == 1) & (df['forward_step'] > 0.1)]

            if not valid_increases.empty:
                all_increases.extend(valid_increases['forward_step'].values)
        except:
            pass

    if not all_increases:
        print("Warning: No positive steps found. Defaulting to 10.0")
        return 10.0

    avg_step = np.mean(all_increases)
    print(f">>> Calculated Average Step Size: {avg_step:.4f}")
    return avg_step


# ==========================================
# 3. Data Loading and Sequence Construction
# ==========================================

def read_folder_csvs(folder_path):
    """Reads the csv files from a folder into a list of dataframes."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            print(df.shape)
            df.attrs['source_file'] = os.path.basename(filename)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

        print(len(dfs))
    return dfs


def process_single_df_to_sequences(df, scaler, sequence_length,
                                   missing_value=0.0,
                                   scale_sensors=True):
    """
    Reads the time column and removes any duplicates, then reindexes onto the
    dominant time step so gaps become NaN rows.
    If VoltageChange column has NaNs introduced after time handling, those rows will be skipped later.
    If VoltageChange column has '-1', those will be mapped to '0' and the next 5-steps will also be read as '0'.
    Uses normalized values for inputs of current, pressure, radiation.

    Inputs: DataFrame, fitted scaler, sequence length.
    Outputs: X_sequences (N, sequence_length, 4), y_sequences (N, 1).

    Used by: prepare_all_data(), polgun_test.py
    """
    df = df.copy()

    time_col = next((col for col in df.columns if col.lower() == 'time'), None)

    if time_col:
        # Time is either numeric strings ("1", "2") or datetime strings.
        is_numeric = False
        try:
            df[time_col] = pd.to_numeric(df[time_col])
            is_numeric = True
        except ValueError:
            pass

        if not is_numeric:
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                start_time = df[time_col].min()
                df[time_col] = (df[time_col] - start_time).dt.total_seconds()
            except Exception as e:
                print(f"Warning: Could not convert '{time_col}' to numbers. Skipping resampling. Error: {e}")
                time_col = None

    if time_col:
        df = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col)

        if len(df) > 1:
            time_diffs = df[time_col].diff()
            freq = time_diffs.mode()[0]

            if pd.notna(freq) and freq > 0:
                min_t, max_t = df[time_col].min(), df[time_col].max()
                full_time_index = np.arange(min_t, max_t + freq/1000.0, freq)

                # Rounding avoids float precision mismatch (e.g. 1.0000001 vs 1.0)
                df[time_col] = df[time_col].round(6)
                full_time_index = np.round(full_time_index, 6)

                df = df.set_index(time_col).reindex(full_time_index).reset_index()
                df = df.rename(columns={'index': time_col, time_col: time_col})

    # Map target -1 -> 0, and enforce a 5-step refractory window after each -1
    if 'VoltageChange' in df.columns:
        vc_raw = pd.to_numeric(df['VoltageChange'], errors='coerce').values.copy()
        n = len(vc_raw)

        force_zero = np.zeros(n, dtype=bool)
        neg_indices = np.where(vc_raw == -1)[0]
        for idx in neg_indices:
            end = min(n, idx + 6)  # idx, idx+1, ..., idx+5
            force_zero[idx:end] = True

        vc_clean = vc_raw.copy()
        vc_clean[vc_clean == -1] = 0
        vc_clean[force_zero] = 0

        df['VoltageChange'] = vc_clean

    # Channels: [Current, Pressure, Radiation, Prev_A1] (voltage is not an input)
    input_data = []
    for col in FEATURE_COLS:
        # NaN kept as NaN here; replaced with missing_value after normalization
        val = df[col].astype(float)
        input_data.append(val.values)

    a1_history = df['VoltageChange'].fillna(0).values
    input_data.append(a1_history)

    X_raw = np.stack(input_data, axis=1)

    # When scale_sensors=False the scaler is fit downstream (post train/val split)
    # and applied to the windowed sequences via _apply_scaler_to_sensor_channels;
    # NaN values are preserved here so they survive into the windowed array.
    if scale_sensors:
        if scaler is None:
            raise ValueError("scale_sensors=True requires a fitted scaler.")
        nan_mask = np.isnan(X_raw[:, [0, 1, 2]])
        X_sensor_temp = np.nan_to_num(X_raw[:, [0, 1, 2]], nan=0.0)
        X_raw[:, [0, 1, 2]] = scaler.transform(X_sensor_temp)
        for i in range(3):
            X_raw[nan_mask[:, i], i] = missing_value

    if len(X_raw) <= sequence_length:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1))

    X_seq, y_seq = [], []
    y_raw_target = df['VoltageChange'].values

    for i in range(len(X_raw) - sequence_length):
        idx = i + sequence_length
        target_val = y_raw_target[idx]

        # Only keep windows with a valid target (not a filled time gap)
        if np.isnan(target_val):
            continue

        X_seq.append(X_raw[i : i + sequence_length])
        y_seq.append(target_val)

    if len(X_seq) == 0:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1))

    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)


def _apply_scaler_to_sensor_channels(X, scaler, missing_value=0.0):
    """Apply ``scaler`` to the first three sensor channels of ``X`` while
    preserving NaN positions as ``missing_value`` after scaling.

    Args:
        X: array of shape (N, seq_len, 4); only channels 0/1/2 are scaled.
        scaler: fitted StandardScaler with three input features.
        missing_value: value used to mark previously-NaN sensor samples.

    Returns: a copy of ``X`` with sensor channels standardised.
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


def prepare_all_data(train_folder, sequence_length,
                     missing_value=0.0,
                     split_seed=None):
    """
    WHAT: Orchestrates data loading, splitting, and scaling.

    Order of operations (no train -> val scaler leakage):
      1. Read CSV files from the train folder.
      2. Build sequences for every training file with raw, unscaled sensor
         values (NaN preserved).
      3. Shuffle-split the training sequences into train/val (85/15).
      4. Fit a NaN-aware StandardScaler ONLY on the training-portion
         sensor values (channels 0, 1, 2).
      5. Apply that scaler to the train and val sequences and replace NaN
         entries with ``missing_value``.

    Outputs: data dict (X_train, y_train, X_val, y_val), pos_weight, scaler

    X_* are torch FloatTensors of shape (Samples, Seq_Len, 4)
    y_* are torch FloatTensors of shape (Samples, 1) (target: VoltageChange)
    """
    print(f"Loading Data...")

    train_dfs = read_folder_csvs(train_folder)
    print(f"Loaded {len(train_dfs)} training files")

    print("Processing Train Sequences (unscaled, NaN preserved)...")
    X_train_list, y_train_list = [], []
    for df in train_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler=None, sequence_length=sequence_length,
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

    print(f"Splitting into train/val with shuffled train_test_split (85/15, random_state={split_seed})...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=0.15,
        shuffle=True,
        random_state=split_seed
    )

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

    X_train = _apply_scaler_to_sensor_channels(X_train_raw, scaler, missing_value=missing_value)
    X_val = _apply_scaler_to_sensor_channels(X_val_raw, scaler, missing_value=missing_value)

    # Imbalanced data (mostly Hold): Weight = Negatives / Positives
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float)

    data = {
        'X_train': torch.FloatTensor(X_train), 'y_train': torch.FloatTensor(y_train),
        'X_val': torch.FloatTensor(X_val), 'y_val': torch.FloatTensor(y_val),
    }

    n_pos_val = np.sum(y_val == 1)
    n_neg_val = np.sum(y_val == 0)

    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Training Set:   {data['X_train'].shape}  (Samples, Seq_Len, Channels)")
    print(f"Validation Set: {data['X_val'].shape}")
    print("-" * 50)
    print(f"Shuffled Split: 85% train / 15% val (random_state={split_seed})")
    print("-" * 50)
    print(f"CLASS DISTRIBUTION:")
    print(f"  TRAIN:      Stable(0)={int(n_neg):>6}  |  Increase(1)={int(n_pos):>6}  |  1-ratio={100*n_pos/(n_pos+n_neg):.2f}%")
    print(f"  VALIDATION: Stable(0)={int(n_neg_val):>6}  |  Increase(1)={int(n_pos_val):>6}  |  1-ratio={100*n_pos_val/(n_pos_val+n_neg_val+1e-8):.2f}%")
    print("-" * 50)
    print(f"Calculated Pos_Weight (for Train): {pos_weight.item():.4f}")
    print("="*50 + "\n")

    return data, pos_weight, scaler


# ==========================================
# 4. Model Architectures
# ==========================================

def create_lstm_model(input_dim, hidden_dim):
    lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
    linear = nn.Linear(hidden_dim, 1)
    return {'lstm': lstm, 'linear': linear}


def forward_lstm(models, x):
    # LSTM expects (Batch, Seq, Features); logit from the last time step
    lstm_out, _ = models['lstm'](x)
    return models['linear'](lstm_out[:, -1, :])


def _extract_cnn_features(models, x):
    """Run the convolutional feature extractor and return flattened features.

    Both ``forward_cnn`` (inference) and ``create_cnn_model`` (sizing the
    linear head via a dummy probe) call into it, so the linear input
    dimension is always whatever this function actually produces.
    """
    # PyTorch Conv1d expects [Batch, Channels, Time]
    x = x.permute(0, 2, 1)

    # Block 1
    x = models['conv1'](x)
    x = models['bn1'](x)
    x = F.relu(x)

    # Block 2
    x = models['conv2'](x)
    x = models['bn2'](x)
    x = F.relu(x)
    x = models['pool'](x)

    # Block 3
    x = models['conv3'](x)
    x = models['bn3'](x)
    x = F.relu(x)

    return x.reshape(x.size(0), -1)


def _probe_cnn_feature_dim(feature_layers, input_dim, seq_len):
    """Return the flattened feature count produced by ``_extract_cnn_features``
    for a dummy input of shape ``(2, seq_len, input_dim)``.

    Layers are temporarily switched to ``eval()`` so BatchNorm does not
    update running statistics from the probe; original modes are restored.
    """
    saved_modes = {}
    for name, layer in feature_layers.items():
        if hasattr(layer, 'training'):
            saved_modes[name] = layer.training
        if hasattr(layer, 'eval'):
            layer.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(2, seq_len, input_dim)
            flat = _extract_cnn_features(feature_layers, dummy)
        return int(flat.shape[1])
    finally:
        for name, was_training in saved_modes.items():
            layer = feature_layers[name]
            if was_training and hasattr(layer, 'train'):
                layer.train()


def create_cnn_model(
    input_dim,
    seq_len,
    n_filters1: int = 128,
    n_filters2: int = 256,
    kernel_size: int = 3,
    pool_size: int = 2,
):
    """
    ARCH: [Conv1d -> BN -> ReLU] x2 -> MaxPool -> Conv1d -> BN -> ReLU -> Flatten -> Linear

    Output: All the layers in a dict for easy access.
    """
    # Padding preserves the sequence length for odd kernel sizes.
    conv1 = nn.Conv1d(input_dim, n_filters1, kernel_size=kernel_size, padding=kernel_size // 2)
    bn1 = nn.BatchNorm1d(n_filters1)

    conv2 = nn.Conv1d(n_filters1, n_filters1, kernel_size=kernel_size, padding=kernel_size // 2)
    bn2 = nn.BatchNorm1d(n_filters1)

    pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    conv3 = nn.Conv1d(n_filters1, n_filters2, kernel_size=kernel_size, padding=kernel_size // 2)
    bn3 = nn.BatchNorm1d(n_filters2)

    feature_layers = {
        'conv1': conv1, 'bn1': bn1,
        'conv2': conv2, 'bn2': bn2,
        'conv3': conv3, 'bn3': bn3,
        'pool': pool,
    }

    linear_input_size = _probe_cnn_feature_dim(feature_layers, input_dim, seq_len)
    linear = nn.Linear(linear_input_size, 1)

    return {
        'conv1': conv1, 'bn1': bn1,
        'conv2': conv2, 'bn2': bn2,
        'conv3': conv3, 'bn3': bn3,
        'pool': pool,
        'linear': linear,
    }


def forward_cnn(models, x):
    return models['linear'](_extract_cnn_features(models, x))


def summarize_model_parameters(models, model_type="cnn"):
    """Print all trainable parameters for the dict-based model."""
    print("\n" + "="*60)
    print(f"MODEL PARAMETER SUMMARY [{model_type.upper()}]")
    print("="*60)

    total_params = 0

    for layer_name, layer in models.items():
        layer_trainable = 0
        print(f"\nLayer: {layer_name} ({layer.__class__.__name__})")

        for pname, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            num = p.numel()
            layer_trainable += num
            total_params += num
            print(f"  - {pname:20s} shape={tuple(p.shape)}  params={num}")

        if layer_trainable == 0:
            print("  (no trainable parameters)")
        else:
            print(f"  Trainable params in {layer_name}: {layer_trainable}")

    print("\n" + "-"*60)
    print(f"TOTAL TRAINABLE PARAMETERS: {total_params}")
    print("="*60 + "\n")


# ==========================================
# 5. Training Loop
# ==========================================

def train_model(
    model_type,
    data,
    pos_weight,
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
    n_filters1: int = 128,
    n_filters2: int = 256,
    kernel_size: int = 3,
    pool_size: int = 2,
    hidden_dim: int = 64,
    early_stopping_patience: int = 10,
):
    """
    Creates the LSTM or CNN model based on model_type and trains it with Adam
    and BCEWithLogitsLoss(pos_weight). Early stopping monitors validation loss;
    the best-epoch weights are restored before returning.

    Inputs: model_type ('cnn' or 'lstm'), data dict, pos_weight tensor, hyperparameters
    Outputs: trained models dict, forward function, training history dict
    """
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[2]
    print(f"Input dimension: {input_dim}")
    seq_len = X_train.shape[1]

    if model_type == 'lstm':
        print("Initializing LSTM Model...")
        models = create_lstm_model(input_dim, hidden_dim=hidden_dim)
        forward_fn = forward_lstm
    else:
        print("Initializing CNN Model...")
        models = create_cnn_model(
            input_dim,
            seq_len,
            n_filters1=n_filters1,
            n_filters2=n_filters2,
            kernel_size=kernel_size,
            pool_size=pool_size,
        )
        forward_fn = forward_cnn

    params = [p for layer in models.values() for p in layer.parameters()]
    optimizer = optim.Adam(params, lr=learning_rate)

    summarize_model_parameters(models, model_type=model_type)

    # pos_weight scales the loss for the minority class (Increase).
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = copy.deepcopy({name: layer.state_dict() for name, layer in models.items()})

    for epoch in range(epochs):

        # --- TRAIN LOOP (Mini-Batches) ---
        for layer in models.values():
            layer.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = forward_fn(models, batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            n_batch = batch_X.size(0)
            epoch_loss += loss.item() * n_batch
            epoch_correct += ((preds.detach() > 0).float() == batch_y).float().sum().item()
            epoch_total += n_batch

        n_train = len(train_loader.dataset)
        avg_train_loss = epoch_loss / n_train
        train_acc = epoch_correct / epoch_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # --- VALIDATION (eval mode) ---
        for layer in models.values():
            layer.eval()

        with torch.inference_mode():
            val_preds = forward_fn(models, X_val)
            v_loss = criterion(val_preds, y_val)
            val_acc = ((val_preds > 0).float() == y_val).float().mean().item()

            history['val_loss'].append(v_loss.item())
            history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss {avg_train_loss:.4f} | Val Loss {v_loss.item():.4f} | "
            f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
        )

        # --- EARLY STOPPING CHECK (validation loss) ---
        current_val_loss = v_loss.item()
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy({name: layer.state_dict() for name, layer in models.items()})
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n>>> Early stopping triggered at epoch {epoch+1}. "
                      f"No val_loss improvement for {early_stopping_patience} epochs.")
                break

    # --- RESTORE BEST-EPOCH WEIGHTS ---
    for name, layer in models.items():
        layer.load_state_dict(best_state_dict[name])
    print(f">>> Restored best weights from epoch {best_epoch+1} (val_loss={best_val_loss:.6f}).")
    print(f">>> Training summary: best_epoch={best_epoch+1}, best_val_loss={best_val_loss:.6f}, "
          f"last_epoch_run={len(history['train_loss'])}, stopped_early={epochs_without_improvement >= early_stopping_patience}")

    return models, forward_fn, history


# ==========================================
# 6. Model Checkpointing (Save & Load)
# ==========================================

def save_checkpoint(output_folder, models, scaler, hyperparams, history,
                    pos_weight, train_dir, test_dir, avg_step,
                    channel_names, model_type='cnn', val_threshold_selection=None):
    """
    Saves both lightweight and full checkpoints.

    Lightweight (checkpoint.pth): Model weights only - fast loading for inference
    Full (full_checkpoint.pth): Everything needed for testing and reproducibility,
    including the validation-selected decision threshold t*.
    """
    lightweight_path = os.path.join(output_folder, "checkpoint.pth")
    torch.save(
        {name: layer.state_dict() for name, layer in models.items()},
        lightweight_path
    )
    print(f"Saved lightweight checkpoint: {lightweight_path}")

    full_checkpoint = {
        # Model
        'model_state_dict': {name: layer.state_dict() for name, layer in models.items()},
        'model_type': model_type,

        # Preprocessing (scaler parameters for reconstructing StandardScaler)
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'scaler_var': scaler.var_.tolist(),
        'scaler_n_features': scaler.n_features_in_,

        # Hyperparameters
        'hyperparameters': hyperparams,

        # Training results
        'training_history': history,
        'val_threshold_selection': val_threshold_selection,

        # Data info
        'pos_weight': pos_weight.item() if torch.is_tensor(pos_weight) else pos_weight,
        'train_dir': train_dir,
        'test_dir': test_dir,
        'channel_names': channel_names,

        # Constraint learned from data
        'avg_step_size': avg_step,
    }

    full_path = os.path.join(output_folder, "full_checkpoint.pth")
    torch.save(full_checkpoint, full_path)
    print(f"Saved full checkpoint: {full_path}")

    return lightweight_path, full_path


def load_checkpoint(checkpoint_path, model_type='cnn'):
    """
    Loads a saved checkpoint and reconstructs model + scaler.

    Args:
        checkpoint_path: Path to full_checkpoint.pth
        model_type: 'cnn' or 'lstm' (overridden by the checkpoint's model_type)

    Returns:
        models: Reconstructed model dict with loaded weights
        forward_fn: Forward function for the model
        scaler: Reconstructed StandardScaler
        checkpoint: Full checkpoint dict with all saved data
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # weights_only=False: full_checkpoint.pth stores scaler params, history,
    # numpy scalars, etc. Required for PyTorch >= 2.6.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    scaler.var_ = np.array(checkpoint['scaler_var'])
    scaler.n_features_in_ = checkpoint['scaler_n_features']

    hyperparams = checkpoint['hyperparameters']
    seq_len = hyperparams['sequence_length']
    input_dim = 4  # 3 sensors + Prev_A1

    model_type = checkpoint.get('model_type', model_type)

    if model_type == 'lstm':
        models = create_lstm_model(
            input_dim,
            hidden_dim=hyperparams.get('hidden_dim', 64),
        )
        forward_fn = forward_lstm
    else:
        models = create_cnn_model(
            input_dim,
            seq_len,
            n_filters1=hyperparams.get('n_filters1', 128),
            n_filters2=hyperparams.get('n_filters2', 256),
            kernel_size=hyperparams.get('kernel_size', 3),
            pool_size=hyperparams.get('pool_size', 2),
        )
        forward_fn = forward_cnn

    for name, state_dict in checkpoint['model_state_dict'].items():
        if name in models:
            models[name].load_state_dict(state_dict)

    for layer in models.values():
        layer.eval()

    print(">>> Checkpoint loaded successfully!")
    print(f"    Model type: {model_type}")
    print(f"    Epochs trained: {len(checkpoint['training_history']['train_loss'])}")
    print(f"    Sequence length: {seq_len}")

    return models, forward_fn, scaler, checkpoint


# ==========================================
# 7. Training Plots and Summary
# ==========================================

def plot_training_curves(history, output_folder=None):
    """Training and validation loss versus epoch."""
    epochs = range(1, len(history['val_loss']) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history['train_loss'], label='Training')
    ax.plot(epochs, history['val_loss'], label='Validation', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_folder:
        save_figure(fig, output_folder, "training_curves")
    plt.close(fig)


def save_training_summary_txt(output_folder, history, threshold_selection, models, hyperparams, model_type, data):
    """
    Writes a combined training_summary.txt with:
    1) Model architecture summary
    2) Validation metrics at the committed threshold t*
    3) Early stopping summary
    4) Per-epoch table (loss, accuracy)
    """
    lines = []

    # --- Section 1: Architecture ---
    seq_len = data['X_train'].shape[1]
    input_channels = data['X_train'].shape[2]
    lines.append("=" * 90)
    lines.append("MODEL ARCHITECTURE")
    lines.append("=" * 90)
    lines.append(f"Model type:      {model_type}")
    lines.append(f"Input channels:  {input_channels}")
    lines.append(f"Sequence length: {seq_len}")
    lines.append("")
    total_params = 0
    for layer_name, layer in models.items():
        layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total_params += layer_params
        lines.append(f"  {layer_name:20s} | {layer.__class__.__name__:20s} | params: {layer_params}")
    lines.append(f"\n  TOTAL TRAINABLE PARAMS: {total_params}")
    lines.append("")
    lines.append("HYPERPARAMETERS:")
    for k, v in hyperparams.items():
        lines.append(f"  {k}: {v}")

    # --- Section 2: Validation metrics at t* ---
    lines.append("")
    lines.append("=" * 90)
    lines.append("VALIDATION METRICS AT COMMITTED THRESHOLD")
    lines.append("=" * 90)
    if threshold_selection:
        chosen = threshold_selection.get('chosen_row', {})
        lines.append(f"  {'criterion':25s}: {threshold_selection.get('criterion')}")
        lines.append(f"  {'chosen_threshold':25s}: {threshold_selection.get('chosen_threshold'):.6f}")
        for k in ('precision', 'recall', 'f1', 'mcc'):
            lines.append(f"  {k:25s}: {chosen.get(k, float('nan')):.6f}")
        lines.append(f"  {'average_precision':25s}: {threshold_selection.get('average_precision'):.6f}")
    else:
        lines.append("  (no threshold selection available)")

    # --- Section 3: Early stopping summary ---
    n_epochs = len(history['train_loss'])
    best_epoch = int(np.argmin(history['val_loss']))
    best_val_loss = history['val_loss'][best_epoch]
    lines.append("")
    lines.append("=" * 90)
    lines.append("TRAINING SUMMARY")
    lines.append("=" * 90)
    lines.append(f"  Epochs trained:    {n_epochs}")
    lines.append(f"  Best epoch:        {best_epoch + 1}")
    lines.append(f"  Best val_loss:     {best_val_loss:.6f}")

    # --- Section 4: Per-epoch table ---
    lines.append("")
    lines.append("=" * 90)
    lines.append("PER-EPOCH HISTORY")
    lines.append("=" * 90)

    header = (f"{'epoch':>5s} | {'tr_loss':>8s} | {'va_loss':>8s} | "
              f"{'tr_acc':>7s} | {'va_acc':>7s}")
    lines.append(header)
    lines.append("-" * len(header))

    def _g(key, idx):
        vals = history.get(key, [])
        return vals[idx] if idx < len(vals) and vals[idx] is not None else float('nan')

    for i in range(n_epochs):
        row = (f"{i+1:5d} | {_g('train_loss',i):8.4f} | {_g('val_loss',i):8.4f} | "
               f"{_g('train_acc',i):7.4f} | {_g('val_acc',i):7.4f}")
        lines.append(row)

    content = "\n".join(lines)
    save_text_report(output_folder, "training_summary", content)


# ==========================================
# 8. Multi-Seed Validation Aggregation
# ==========================================

VAL_METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "pr_auc", "mcc"]


def _derive_accuracy(chosen_row, class_counts):
    """Reconstruct validation accuracy at the chosen threshold from saved fields.

    precision = TP / predicted_positive_count, recall = TP / n_pos, and
    validation_class_counts give n_pos / n_neg -> TP, FP, FN, TN -> accuracy.
    """
    n_pos = int(class_counts.get("increase_1", 0))
    n_neg = int(class_counts.get("stable_0", 0))
    n_total = n_pos + n_neg
    if n_total == 0:
        return float("nan")

    precision = float(chosen_row.get("precision", 0.0))
    recall = float(chosen_row.get("recall", 0.0))
    pp = int(chosen_row.get("predicted_positive_count", 0))

    if n_pos > 0:
        tp_float = recall * n_pos
    elif pp > 0:
        tp_float = precision * pp
    else:
        tp_float = 0.0

    tp = int(round(tp_float))
    tp = max(0, min(tp, n_pos, pp))
    fp = max(pp - tp, 0)
    tn = max(n_neg - fp, 0)
    return (tp + tn) / n_total


def aggregate_multi_seed_validation_metrics(root_output_folder, run_records):
    """Save per-run validation metrics and mean/SD across the seed grid.

    Metrics are read from each run's ``val_threshold_selection.json`` at the
    validation-committed t* (``chosen_row``); PR-AUC is the validation average
    precision. Writes ``all_runs_validation_metrics.csv`` and
    ``aggregate_validation_metrics_mean_sd.csv`` / ``.json``.
    """
    rows = []
    for record in run_records:
        val_path = os.path.join(record["output_folder"], "val_threshold_selection.json")
        row = {
            "run_index": record["run_index"],
            "weight_init_seed": record["weight_init_seed"],
            "split_seed": record["split_seed"],
            "output_folder": record["output_folder"],
            "status": "missing_val_threshold_selection",
            "chosen_threshold": np.nan,
            "criterion": None,
        }
        for m in VAL_METRIC_NAMES:
            row[m] = np.nan

        if os.path.exists(val_path):
            try:
                with open(val_path, "r", encoding="utf-8") as f:
                    sel = json.load(f)
                chosen = sel.get("chosen_row") or {}
                class_counts = sel.get("validation_class_counts") or {}
                row["status"] = "ok"
                row["criterion"] = sel.get("criterion")
                row["chosen_threshold"] = float(sel.get("chosen_threshold", np.nan))
                row["accuracy"] = _derive_accuracy(chosen, class_counts)
                row["precision"] = float(chosen.get("precision", np.nan))
                row["recall"] = float(chosen.get("recall", np.nan))
                row["f1"] = float(chosen.get("f1", np.nan))
                row["mcc"] = float(chosen.get("mcc", np.nan))
                row["pr_auc"] = float(sel.get("average_precision", np.nan))
            except Exception as exc:
                row["status"] = f"read_error: {exc}"

        rows.append(row)

    all_df = pd.DataFrame(rows)
    all_csv = os.path.join(root_output_folder, "all_runs_validation_metrics.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved: {all_csv}")

    ok_df = all_df[all_df["status"] == "ok"]

    def _stats(values):
        return {
            "n_runs": int(values.shape[0]),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()) if len(values) else float("nan"),
            "max": float(values.max()) if len(values) else float("nan"),
            "median": float(values.median()) if len(values) else float("nan"),
        }

    summary_rows = []
    for metric_name in VAL_METRIC_NAMES:
        values = pd.to_numeric(ok_df[metric_name], errors="coerce").dropna()
        summary_rows.append({"metric": metric_name, **_stats(values)})

    chosen_t = pd.to_numeric(ok_df["chosen_threshold"], errors="coerce").dropna()
    threshold_stats = _stats(chosen_t)
    summary_rows.append({"metric": "chosen_threshold", **threshold_stats})

    criteria_used = sorted({c for c in ok_df.get("criterion", []) if c})

    summary_csv = os.path.join(root_output_folder, "aggregate_validation_metrics_mean_sd.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    summary_json = os.path.join(root_output_folder, "aggregate_validation_metrics_mean_sd.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "n_expected_runs": len(run_records),
            "n_completed_runs": int(len(ok_df)),
            "metrics_source": (
                "val_threshold_selection.json -> chosen_row "
                "(metrics at validation-committed t*)"
            ),
            "accuracy_derivation": (
                "Reconstructed from chosen_row.precision, chosen_row.recall, "
                "chosen_row.predicted_positive_count, and validation_class_counts."
            ),
            "threshold_criteria_used": criteria_used,
            "metrics": summary_rows,
            "chosen_threshold_stats": threshold_stats,
        }, f, indent=2)
    print(f"Saved: {summary_json}")

    print("\nOverall VALIDATION metrics across completed runs (mean +/- SD, at chosen t*):")
    for r in summary_rows:
        print(f"  {r['metric']:>16s}: {r['mean']:.4f} +/- {r['sd']:.4f} (n={r['n_runs']})")
    if criteria_used:
        print(f"Threshold criteria observed in runs: {criteria_used}")

    missing = all_df[all_df["status"] != "ok"]
    if len(missing) > 0:
        print(f"\nWARNING: {len(missing)} run(s) skipped (missing/unreadable):")
        for _, row in missing.iterrows():
            print(f"  - run_{int(row['run_index']):02d} ({row['output_folder']}): {row['status']}")


def run_multi_seed_experiments():
    """Launch 25 isolated single-run processes and aggregate the validation metrics."""
    root_output_folder = os.path.abspath(create_output_folder(prefix=f"polgun_{MODEL_TYPE}_multiseed"))
    run_records = []
    total_runs = len(WEIGHT_INIT_SEEDS) * len(SPLIT_SEEDS)

    print("\n" + "=" * 80)
    print(f"STARTING MULTI-SEED EXPERIMENT GRID: {total_runs} RUNS")
    print(f"Model type:        {MODEL_TYPE}")
    print(f"Weight-init seeds: {WEIGHT_INIT_SEEDS}")
    print(f"Split seeds:       {SPLIT_SEEDS}")
    print(f"Root output:       {root_output_folder}")
    print("=" * 80 + "\n")

    run_index = 0
    for weight_seed in WEIGHT_INIT_SEEDS:
        for split_seed in SPLIT_SEEDS:
            run_index += 1
            run_name = f"run_{run_index:02d}_init_{weight_seed}_split_{split_seed}"
            run_output_folder = os.path.join(root_output_folder, run_name)
            os.makedirs(os.path.join(run_output_folder, "figures"), exist_ok=True)

            print("\n" + "=" * 80)
            print(f"RUN {run_index}/{total_runs}: weight_init_seed={weight_seed}, split_seed={split_seed}")
            print(f"Output folder: {run_output_folder}")
            print("=" * 80)

            env = os.environ.copy()
            env["POLGUN_SINGLE_RUN"] = "1"
            env["POLGUN_WEIGHT_INIT_SEED"] = str(weight_seed)
            env["POLGUN_SPLIT_SEED"] = str(split_seed)
            env["POLGUN_OUTPUT_FOLDER"] = run_output_folder
            env["PYTHONUNBUFFERED"] = "1"

            subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, check=True)
            run_records.append({
                "run_index": run_index,
                "weight_init_seed": weight_seed,
                "split_seed": split_seed,
                "output_folder": run_output_folder,
            })

    aggregate_multi_seed_validation_metrics(root_output_folder, run_records)
    print("\nAll multi-seed runs complete.")


# ==========================================
# 9. Single Training Run
# ==========================================

def run_single_training():
    """Train one model for the seeds given by POLGUN_WEIGHT_INIT_SEED / POLGUN_SPLIT_SEED."""
    weight_init_seed = int(os.environ.get("POLGUN_WEIGHT_INIT_SEED", "545"))
    split_seed = int(os.environ.get("POLGUN_SPLIT_SEED", "97"))
    set_all_seeds(weight_init_seed)
    print(f"\n>>> Single-run seeds: weight_init_seed={weight_init_seed}, split_seed={split_seed}")

    hyperparams = dict(HYPERPARAMS)
    hyperparams['weight_init_seed'] = weight_init_seed
    hyperparams['split_seed'] = split_seed
    seq_len = hyperparams['sequence_length']

    output_folder = create_output_folder(prefix=f"polgun_{MODEL_TYPE}")

    # 1. Average step size learned from the training data (used by the test simulation)
    avg_step_size = calculate_average_step_size(TRAIN_DIR)

    # 2. Data: shuffled train/val split, NaN-aware normalization
    data, pos_weight, scaler = prepare_all_data(
        TRAIN_DIR, seq_len,
        missing_value=hyperparams['missing_value'],
        split_seed=split_seed,
    )

    # 3. Train
    system_specs = get_system_specs()
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_start_time = time.time()

    models, fwd_fn, history = train_model(
        MODEL_TYPE, data, pos_weight,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['learning_rate'],
        n_filters1=hyperparams['n_filters1'],
        n_filters2=hyperparams['n_filters2'],
        kernel_size=hyperparams['kernel_size'],
        pool_size=hyperparams['pool_size'],
        hidden_dim=hyperparams['hidden_dim'],
        early_stopping_patience=hyperparams['early_stopping_patience'],
    )

    training_seconds = time.time() - train_start_time
    timing_report = save_system_and_timing(output_folder, system_specs, training_seconds, run_timestamp)

    # 4. Training curves and per-epoch history
    plot_training_curves(history, output_folder=output_folder)
    save_epoch_history_report(output_folder, history)

    # 5. Validation threshold selection (t* used by polgun_test.py)
    chosen_threshold, threshold_selection = compute_validation_pr_threshold(
        models, fwd_fn, data,
        output_folder=output_folder,
        criterion=hyperparams.get('threshold_criterion', 'max_mcc'),
    )

    # 6. Reports
    save_model_architecture_report(output_folder, models, hyperparams, MODEL_TYPE, data)
    save_training_summary_txt(output_folder, history, threshold_selection, models,
                              hyperparams, MODEL_TYPE, data)

    # 7. Checkpoint
    print("\n>>> Saving model checkpoint...")
    save_checkpoint(
        output_folder=output_folder,
        models=models,
        scaler=scaler,
        hyperparams=hyperparams,
        history=history,
        pos_weight=pos_weight,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        avg_step=avg_step_size,
        channel_names=CHANNEL_NAMES,
        model_type=MODEL_TYPE,
        val_threshold_selection=threshold_selection,
    )

    print("\n" + "="*70)
    print("RUN COMPLETE - OUTPUT SUMMARY")
    print("="*70)
    print(f"All outputs saved to: {output_folder}/")
    print(f"  - figures/training_curves.png, figures/val_mcc_vs_threshold.png")
    print(f"  - epoch_history.json, training_summary.txt, model_architecture.json")
    print(f"  - val_threshold_selection.json (t* = {chosen_threshold:.6f}), val_pr_curve_points.csv")
    print(f"  - system_and_timing.json / .txt")
    print(f"  - checkpoint.pth (weights only), full_checkpoint.pth (complete)")
    print(f"\nTraining time: {timing_report['training_wall_clock_formatted']} ({training_seconds:.1f}s)")
    print("="*70)


if __name__ == "__main__":
    if os.environ.get("POLGUN_SINGLE_RUN") != "1":
        run_multi_seed_experiments()
        sys.exit(0)
    run_single_training()
