"""
polgun_test.py
==============

Test-set evaluation and plots for models trained with ``polgun_train.py``.
Nothing is retrained and the decision threshold is never re-selected: each
run's validation-committed t* is applied to the test data.

Inputs (asked in the terminal when not given on the command line)
------
1. Checkpoint path, one of:
     - a multi-seed mother folder (every ``run_*`` subfolder is evaluated),
     - a single run folder, or
     - a ``full_checkpoint.pth`` file.
   Model type, hyperparameters, scaler, avg_step_size, and t* are read from
   the checkpoint (t* falls back to ``val_threshold_selection.json``).
2. Test CSV file or folder of CSVs. Press Enter to use the ``test_dir`` stored
   in the checkpoint.
3. Optional second checkpoint path holding the other model type (CNN <-> LSTM)
   for the combined Real + CNN + LSTM voltage plot. Runs are matched by run
   folder name. Press Enter to skip.

Outputs (in ``<mother or run folder>/test_<test name>/``)
-------
Per run (``<run name>/``):
    test_metrics.json, test_metrics_by_file.csv,
    figures/ (rescaled simulated vs real voltage PDF + PNG)
Across runs:
    all_runs_test_metrics.csv, aggregate_test_metrics_mean_sd.csv / .json

Usage
-----
    python polgun_test.py
    python polgun_test.py --checkpoint "polgun_cnn_multiseed_..." --test "test poster"
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_score, recall_score)

from polgun_train import load_checkpoint, process_single_df_to_sequences


VOLTAGE_COL_PRIMARY = "glassmanDataXfer:hvPsVoltageMeasM"
VOLTAGE_COL_FALLBACK = "hvps.lerec:voltageM"

COLOR_REAL = "#2166AC"
COLOR_CNN = "#D95F02"
COLOR_LSTM = "#1B7837"

MODEL_STYLE = {
    "CNN": {"color": COLOR_CNN, "marker": "o", "markersize": 5.5},
    "LSTM": {"color": COLOR_LSTM, "marker": "^", "markersize": 6.0},
}


# ==========================================
# 1. Inputs: checkpoints, test files, threshold
# ==========================================

def _prompt_path(prompt, allow_empty=False):
    """Ask for a path until an existing file/folder is given (or Enter if allowed)."""
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            if allow_empty:
                return None
            print("Path cannot be empty. Try again.\n")
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.exists(path):
            return path
        print(f"Not found: {path}\n")


def _discover_checkpoints(path):
    """Return (base_folder, [(run_name, checkpoint_path), ...]) for a file, run folder, or mother folder."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        run_folder = os.path.dirname(path)
        return run_folder, [(os.path.basename(run_folder), path)]

    own_ckpt = os.path.join(path, "full_checkpoint.pth")
    if os.path.exists(own_ckpt):
        return path, [(os.path.basename(path), own_ckpt)]

    runs = []
    for name in sorted(os.listdir(path)):
        ckpt = os.path.join(path, name, "full_checkpoint.pth")
        if name.startswith("run_") and os.path.exists(ckpt):
            runs.append((name, ckpt))
    if not runs:
        raise RuntimeError(f"No run_*/full_checkpoint.pth found under: {path}")
    return path, runs


def _discover_test_csvs(path):
    """Return the sorted list of CSV files for a CSV path or a folder of CSVs."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSV files found in: {path}")
    return files


def _resolve_threshold(checkpoint, checkpoint_path):
    """Return (t*, source). Checkpoint first, then val_threshold_selection.json; no 0.5 fallback."""
    selection = checkpoint.get("val_threshold_selection") or {}
    if selection.get("chosen_threshold") is not None:
        return float(selection["chosen_threshold"]), f"checkpoint, criterion={selection.get('criterion')}"

    val_path = os.path.join(os.path.dirname(checkpoint_path), "val_threshold_selection.json")
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("chosen_threshold") is not None:
            return float(data["chosen_threshold"]), f"val_threshold_selection.json, criterion={data.get('criterion')}"

    raise RuntimeError(
        f"No validation threshold found for {checkpoint_path}: neither the checkpoint "
        f"nor val_threshold_selection.json contains chosen_threshold."
    )


def _print_diagnostics(checkpoint, threshold, threshold_source, label=None):
    """Print detected model + hyperparameters + threshold, as proof of contents."""
    model_type = checkpoint.get("model_type", "unknown")
    state = checkpoint.get("model_state_dict", {})
    layer_names = list(state.keys())
    title = "CHECKPOINT DIAGNOSTICS"
    if label:
        title = f"{label} {title}"

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Detected model type : {model_type}")
    print(f"Layers in state_dict: {', '.join(layer_names) if layer_names else '(none)'}")

    hp = checkpoint.get("hyperparameters", {}) or {}
    print("\nHyperparameters:")
    if hp:
        width = max(len(str(k)) for k in hp.keys())
        for k in sorted(hp.keys()):
            print(f"  {str(k):<{width}} : {hp[k]}")
    else:
        print("  (no hyperparameters stored)")

    print("\nOther saved facts:")
    print(f"  avg_step_size      : {checkpoint.get('avg_step_size')}")
    print(f"  scaler_n_features  : {checkpoint.get('scaler_n_features')}")
    hist = checkpoint.get("training_history", {}) or {}
    n_epochs = len(hist.get("train_loss", [])) if isinstance(hist, dict) else None
    print(f"  epochs trained     : {n_epochs}")
    channels = checkpoint.get("channel_names")
    if channels:
        print(f"  channel_names      : {channels}")

    print("\nDecision threshold:")
    print(f"  >>> t* = {threshold:.6f}  ({threshold_source})")
    print("=" * 70 + "\n")


# ==========================================
# 2. Test Metrics
# ==========================================

def _compute_classification_metrics(probs, y_true, threshold, include_pr_auc=True):
    """Compute the standard test metric bundle from probabilities and labels.

    Shared by the overall test evaluation and the per-file test evaluation so
    every reported metric uses identical formulas and the same decision
    ``threshold``. PR-AUC is suppressed when only one class is present, which
    happens often on small per-file slices.
    """
    probs = np.asarray(probs, dtype=float).flatten()
    y_true = np.asarray(y_true, dtype=int).flatten()
    preds = (probs >= threshold).astype(int)

    n = int(len(probs))
    if n == 0:
        return None

    accuracy = float((preds == y_true).mean())
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    pr_auc = None
    if include_pr_auc and len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, probs))

    if preds.sum() in (0, n) or y_true.sum() in (0, n):
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(y_true, preds))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    real_vals, real_counts = np.unique(y_true, return_counts=True)
    pred_vals, pred_counts = np.unique(preds, return_counts=True)
    real_dict = {str(k): int(v) for k, v in zip(real_vals, real_counts)}
    pred_dict = {str(k): int(v) for k, v in zip(pred_vals, pred_counts)}

    return {
        'n_samples': n,
        'test_accuracy': accuracy,
        'decision_threshold': float(threshold),
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'confusion_matrix': cm.tolist(),
        'real_distribution': real_dict,
        'predicted_distribution': pred_dict,
    }


def _compute_per_file_test_metrics(probs, preds, y_true, source_files, threshold):
    """Group test-set predictions by their source CSV and compute one metric
    bundle per file at the validation-committed ``threshold``.

    Returns a dict keyed by filename. Files with no surviving sequences are
    silently skipped; PR-AUC and MCC are reported as ``None``/``0.0`` for
    single-class files (handled inside ``_compute_classification_metrics``).
    """
    probs = np.asarray(probs, dtype=float).flatten()
    y_true = np.asarray(y_true, dtype=int).flatten()
    source_files = np.asarray(source_files, dtype=object).flatten()

    if len(source_files) != len(probs):
        print(
            f"WARNING: per-file metrics skipped -- source_files length "
            f"({len(source_files)}) != probabilities length ({len(probs)})."
        )
        return {}

    per_file = {}
    unique_files = sorted({str(f) for f in source_files})
    for fname in unique_files:
        mask = (source_files == fname)
        bundle = _compute_classification_metrics(
            probs[mask], y_true[mask], threshold, include_pr_auc=True
        )
        if bundle is None:
            continue
        per_file[fname] = bundle
    return per_file


def evaluate_test_set(models, forward_fn, X_test, y_test, source_files, pos_weight, threshold):
    """
    Evaluates the trained model on the test set at the validation-committed
    decision ``threshold`` t*. All reported metrics (precision, recall, F1,
    MCC, confusion matrix, etc.) are computed at that single threshold.

    The returned dict also includes ``per_file_test_metrics`` keyed by source
    CSV filename, using the same threshold and metric helper.
    """
    print("\n" + "="*30)
    print("EVALUATING ON TEST SET")
    print(f"Decision threshold: {threshold:.6f}")
    print("="*30)

    if len(X_test) == 0:
        print("Test set is empty. Cannot evaluate.")
        return {}

    for layer in models.values():
        layer.eval()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.inference_mode():
        logits = forward_fn(models, X_test)
        loss = criterion(logits, y_test)
        test_loss = loss.item()

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        y_true = y_test.cpu().numpy().flatten().astype(int)

    overall = _compute_classification_metrics(probs, y_true, threshold, include_pr_auc=True)
    preds = (probs >= threshold).astype(int)

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {overall['test_accuracy']:.4f}")
    print(f"Precision:     {overall['precision']:.4f}")
    print(f"Recall:        {overall['recall']:.4f}")
    print(f"F1 Score:      {overall['f1_score']:.4f}")
    pr_auc_disp = (f"{overall['pr_auc']:.4f}" if overall['pr_auc'] is not None else "n/a (single-class)")
    print(f"PR-AUC:        {pr_auc_disp}")
    print(f"MCC:           {overall['mcc']:.4f}")
    print("-" * 30)
    print(f"Real Distribution:      {overall['real_distribution']}")
    print(f"Predicted Distribution: {overall['predicted_distribution']}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(np.array(overall['confusion_matrix']))
    print("="*30 + "\n")

    metrics = {
        'test_loss': test_loss,
        'test_accuracy': overall['test_accuracy'],
        'decision_threshold': float(threshold),
        'precision': overall['precision'],
        'recall': overall['recall'],
        'f1_score': overall['f1_score'],
        'pr_auc': overall['pr_auc'],
        'mcc': overall['mcc'],
        'confusion_matrix': overall['confusion_matrix'],
        'real_distribution': overall['real_distribution'],
        'predicted_distribution': overall['predicted_distribution'],
    }

    if source_files is not None and len(source_files) > 0:
        per_file = _compute_per_file_test_metrics(
            probs, preds, y_true, source_files, threshold
        )
        if per_file:
            metrics['per_file_test_metrics'] = per_file
            metrics['_test_source_files'] = [str(f) for f in source_files]
            print(f"Computed per-file test metrics for {len(per_file)} source file(s) "
                  f"at decision threshold {threshold:.6f}.")

    return metrics


def save_test_metrics_report(output_folder, test_metrics, y_test, pos_weight, checkpoint_path, test_files):
    """
    Write test-set evaluation metrics, the test class distribution, and the
    training pos_weight to ``test_metrics.json``.

    The per-file bundles in ``per_file_test_metrics`` are mirrored into
    ``test_metrics_by_file.csv``.
    """
    y_test = y_test.cpu().numpy().flatten()

    public_test_metrics = {
        key: value for key, value in test_metrics.items()
        if not key.startswith("_")
    }

    report = {
        'test_metrics': public_test_metrics,
        'class_distribution': {
            'test': {
                'stable_0': int(np.sum(y_test == 0)),
                'increase_1': int(np.sum(y_test == 1)),
            },
        },
        'pos_weight': float(pos_weight.item()) if torch.is_tensor(pos_weight) else float(pos_weight),
        'checkpoint_path': checkpoint_path,
        'test_files': [os.path.basename(p) for p in test_files],
    }
    filepath = os.path.join(output_folder, "test_metrics.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {filepath}")

    per_file = test_metrics.get('per_file_test_metrics')
    if per_file:
        decision_threshold = test_metrics.get('decision_threshold')
        rows = []
        for fname in sorted(per_file.keys()):
            bundle = per_file[fname]
            real = bundle.get('real_distribution', {}) or {}
            pred = bundle.get('predicted_distribution', {}) or {}
            cm = bundle.get('confusion_matrix') or [[0, 0], [0, 0]]
            tn = int(cm[0][0]); fp = int(cm[0][1])
            fn = int(cm[1][0]); tp = int(cm[1][1])
            rows.append({
                'source_file': fname,
                'decision_threshold': bundle.get('decision_threshold', decision_threshold),
                'n_samples': bundle.get('n_samples'),
                'n_stable_0': int(real.get('0', 0)),
                'n_increase_1': int(real.get('1', 0)),
                'n_pred_0': int(pred.get('0', 0)),
                'n_pred_1': int(pred.get('1', 0)),
                'accuracy': bundle.get('test_accuracy'),
                'precision': bundle.get('precision'),
                'recall': bundle.get('recall'),
                'f1_score': bundle.get('f1_score'),
                'pr_auc': bundle.get('pr_auc'),
                'mcc': bundle.get('mcc'),
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'true_positive': tp,
            })
        csv_path = os.path.join(output_folder, "test_metrics_by_file.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


# ==========================================
# 3. Multi-Run Test Aggregation
# ==========================================

def _get_test_metric_value(report, key):
    """Read one scalar test metric from a saved per-run test_metrics.json report."""
    metrics = report.get("test_metrics", {})
    metric_key_map = {
        "accuracy": "test_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1_score",
        "pr_auc": "pr_auc",
        "mcc": "mcc",
        "decision_threshold": "decision_threshold",
    }
    value = metrics.get(metric_key_map[key])
    return float(value) if value is not None else np.nan


def _read_val_chosen_threshold(run_folder):
    """Return the validation-committed t* for one run, or NaN if unavailable."""
    val_path = os.path.join(run_folder, "val_threshold_selection.json")
    if not os.path.exists(val_path):
        return np.nan, None
    try:
        with open(val_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return np.nan, None
    t = data.get("chosen_threshold")
    crit = data.get("criterion")
    return (float(t) if t is not None else np.nan, crit)


def aggregate_multi_seed_results(test_root_folder, run_records):
    """Save individual run test metrics and overall mean/SD for the seed grid.

    Test metrics (accuracy, precision, recall, F1, PR-AUC, MCC) come from each
    run's ``test_metrics.json`` at its validation-committed decision threshold,
    which is included in the per-run CSV and in the aggregate summary. The
    validation threshold is reported as the mean (and SD) of the per-run
    ``chosen_threshold`` read from each run's ``val_threshold_selection.json``.
    """
    metric_names = ["accuracy", "precision", "recall", "f1", "pr_auc", "mcc",
                    "decision_threshold"]
    rows = []

    for record in run_records:
        metrics_path = os.path.join(record["output_folder"], "test_metrics.json")
        val_t, val_crit = _read_val_chosen_threshold(record["run_folder"])
        row = {
            "run_index": record["run_index"],
            "run_name": record["run_name"],
            "model_type": record["model_type"],
            "weight_init_seed": record["weight_init_seed"],
            "split_seed": record["split_seed"],
            "output_folder": record["output_folder"],
            "status": "missing_metrics",
            "val_chosen_threshold": val_t,
            "val_threshold_criterion": val_crit,
        }

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            row["status"] = "ok"
            for metric_name in metric_names:
                row[metric_name] = _get_test_metric_value(report, metric_name)
        else:
            for metric_name in metric_names:
                row[metric_name] = np.nan

        rows.append(row)

    all_runs_df = pd.DataFrame(rows)
    all_runs_path = os.path.join(test_root_folder, "all_runs_test_metrics.csv")
    all_runs_df.to_csv(all_runs_path, index=False)
    print(f"Saved: {all_runs_path}")

    ok_df = all_runs_df[all_runs_df["status"] == "ok"]
    summary_rows = []
    for metric_name in metric_names:
        values = pd.to_numeric(ok_df[metric_name], errors="coerce").dropna()
        summary_rows.append({
            "metric": metric_name,
            "n_runs": int(values.shape[0]),
            "mean": float(values.mean()) if len(values) else np.nan,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        })

    val_threshold_values = pd.to_numeric(
        all_runs_df["val_chosen_threshold"], errors="coerce"
    ).dropna()
    val_threshold_summary = {
        "metric": "val_chosen_threshold",
        "n_runs": int(val_threshold_values.shape[0]),
        "mean": float(val_threshold_values.mean()) if len(val_threshold_values) else np.nan,
        "sd":   float(val_threshold_values.std(ddof=1)) if len(val_threshold_values) > 1 else 0.0,
    }
    summary_rows.append(val_threshold_summary)

    criteria_used = sorted({c for c in all_runs_df.get("val_threshold_criterion", [])
                            if isinstance(c, str) and c})

    summary_csv_path = os.path.join(test_root_folder, "aggregate_test_metrics_mean_sd.csv")
    summary_json_path = os.path.join(test_root_folder, "aggregate_test_metrics_mean_sd.json")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_expected_runs": len(run_records),
            "n_completed_runs": int((all_runs_df["status"] == "ok").sum()),
            "metrics": summary_rows,
            "val_threshold_criteria_used": criteria_used,
            "notes": (
                "decision_threshold is the per-run test threshold (validation-committed t*) "
                "as stored in each run's test_metrics.json. "
                "val_chosen_threshold is the same t* read from each run's "
                "val_threshold_selection.json; the reported mean is the average across "
                "all completed runs."
            ),
        }, f, indent=2)

    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {summary_json_path}")
    print("\nOverall test metrics across completed runs (mean ± SD):")
    for row in summary_rows:
        if row["metric"] == "val_chosen_threshold":
            continue
        print(f"  {row['metric']}: {row['mean']:.4f} ± {row['sd']:.4f} (n={row['n_runs']})")
    print(
        f"\nValidation chosen_threshold across runs: "
        f"{val_threshold_summary['mean']:.4f} ± {val_threshold_summary['sd']:.4f} "
        f"(n={val_threshold_summary['n_runs']})"
    )
    if criteria_used:
        print(f"Validation threshold criteria observed: {criteria_used}")


# ==========================================
# 4. Simulated vs Real Voltage Plot
# ==========================================

def _simulate_voltage(models, forward_fn, X_seq, threshold, step_size, initial_A=0.0):
    """Run inference at t* and accumulate the simulated voltage ramp (probs >= t*)."""
    X_tensor = torch.FloatTensor(X_seq)

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    decisions = (probs >= threshold).astype(int)

    simulated_A = []
    current_A = float(initial_A)
    for d in decisions:
        if d == 1:
            current_A += step_size
        simulated_A.append(current_A)

    return np.asarray(simulated_A, dtype=float), probs, decisions


def _normalize_01(arr):
    """Min-max rescale an array to [0, 1]; returns zeros if flat/invalid."""
    a = np.asarray(arr, dtype=float)
    a_min = np.nanmin(a)
    a_max = np.nanmax(a)
    if np.isfinite(a_min) and np.isfinite(a_max) and a_max > a_min:
        return (a - a_min) / (a_max - a_min)
    return np.zeros_like(a)


def _seconds_to_mmss(value, _position):
    """Format elapsed seconds as Mins:Secs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes}:{seconds:02d}"


def _elapsed_time_array(df):
    """Return full-file elapsed-seconds array, or None if Time is unavailable."""
    time_col = next((c for c in df.columns if c.lower() == "time"), None)
    if time_col is None:
        return None

    t_dt = pd.to_datetime(df[time_col], errors="coerce")
    if t_dt.notna().sum() > 0:
        return (t_dt - t_dt.min()).dt.total_seconds().to_numpy(dtype=float)

    t_num = pd.to_numeric(df[time_col], errors="coerce")
    if t_num.notna().sum() > 0:
        t = t_num.to_numpy(dtype=float)
        return t - np.nanmin(t)

    return None


def _build_time_axis(df, sequence_length, n_preds, t_elapsed=None):
    """Return (x_axis, x_label, use_mmss).

    Prefer datetime Time column -> elapsed seconds (Mins:Secs ticks), else numeric
    seconds, else plain prediction index. Predictions align with df rows starting
    at ``sequence_length``. When ``t_elapsed`` is provided (shared absolute
    elapsed time for multi-model overlays), segments are NOT re-zeroed so CNN
    and LSTM stay aligned on one axes.
    """
    if t_elapsed is not None:
        seg = t_elapsed[sequence_length: sequence_length + n_preds]
        if len(seg) == n_preds:
            return seg, "Time [Mins:Secs]", True

    time_col = next((c for c in df.columns if c.lower() == "time"), None)
    if time_col is not None:
        t_dt = pd.to_datetime(df[time_col], errors="coerce")
        if t_dt.notna().sum() > 0:
            t = (t_dt - t_dt.min()).dt.total_seconds().to_numpy(dtype=float)
            seg = t[sequence_length: sequence_length + n_preds]
            if len(seg) == n_preds:
                seg = seg - np.nanmin(seg)
                return seg, "Time [Mins:Secs]", True

        t_num = pd.to_numeric(df[time_col], errors="coerce")
        if t_num.notna().sum() > 0:
            t = t_num.to_numpy(dtype=float)
            seg = t[sequence_length: sequence_length + n_preds]
            if len(seg) == n_preds:
                seg = seg - np.nanmin(seg)
                return seg, "Time [Mins:Secs]", True

    return np.arange(n_preds, dtype=float), "Time step", False


def _apply_simulation_style():
    """Matplotlib defaults; PDF fonttype 42 so saved PDFs embed TrueType fonts."""
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _format_axes(ax):
    ax.grid(True, alpha=0.3)


def _save_figure(fig, output_folder, basename):
    """Save PDF + high-resolution PNG (600 dpi)."""
    os.makedirs(output_folder, exist_ok=True)
    saved = []
    for extension in ("pdf", "png"):
        output_path = os.path.join(output_folder, f"{basename}.{extension}")
        fig.savefig(
            output_path,
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
        print(f">>> Saved: {output_path}")
        saved.append(output_path)
    return saved


def _build_real_trace(df, results, t_elapsed):
    """Build Real voltage (x, y_norm) covering the union of all prediction ranges."""
    voltage_col = None
    if VOLTAGE_COL_PRIMARY in df.columns:
        voltage_col = VOLTAGE_COL_PRIMARY
    elif VOLTAGE_COL_FALLBACK in df.columns:
        voltage_col = VOLTAGE_COL_FALLBACK

    if voltage_col is None:
        print("    NOTE: no real voltage column found; plotting models only.")
        return None, None

    real_A = pd.to_numeric(df[voltage_col], errors="coerce").to_numpy(dtype=float)
    start = min(r["sequence_length"] for r in results)
    end = max(r["sequence_length"] + r["n_preds"] for r in results)
    end = min(end, len(real_A))
    real_segment = real_A[start:end]
    if len(real_segment) == 0:
        print("    NOTE: real voltage segment empty; plotting models only.")
        return None, None

    real_norm = _normalize_01(real_segment)
    if t_elapsed is not None and len(t_elapsed) >= end:
        x_real = t_elapsed[start:end]
        return x_real, real_norm

    # Fallback: index axis shared only if all models share the same seq_len
    if len({r["sequence_length"] for r in results}) == 1:
        return np.arange(len(real_norm), dtype=float), real_norm

    print("    NOTE: no shared Time column and differing sequence lengths; "
          "skipping Real overlay.")
    return None, None


def _plot_rescaled_voltage(df, results, output_folder, style):
    """Draw Real + each model's simulated voltage (rescaled 0-1) and save one visual variant.

    results: list of dicts with label ('CNN' or 'LSTM'), sim_norm, sequence_length, n_preds
    style: 'solid' or 'markers'
    """
    t_elapsed = _elapsed_time_array(df)
    x_real, real_norm = _build_real_trace(df, results, t_elapsed)

    _apply_simulation_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    if real_norm is not None and x_real is not None:
        ax.plot(
            x_real, real_norm,
            label="Real",
            color=COLOR_REAL,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )

    x_label, use_mmss = "Time step", False
    for result in results:
        x_model, x_label, use_mmss = _build_time_axis(
            df, result["sequence_length"], result["n_preds"], t_elapsed=t_elapsed
        )
        model_style = MODEL_STYLE[result["label"]]
        kwargs = dict(
            label=f"{result['label']} predicted",
            color=model_style["color"],
            linestyle="-",
            linewidth=2,
        )
        if style == "markers":
            markevery = max(max(result["n_preds"], 1) // 40, 1)
            kwargs.update(marker=model_style["marker"],
                          markersize=model_style["markersize"],
                          markevery=markevery)
        ax.plot(x_model, result["sim_norm"], **kwargs)

    base_name = os.path.splitext(os.path.basename(df.attrs.get("source_csv", "csv")))[0]
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel("Rescaled Voltage", fontsize=20)
    if use_mmss:
        ax.xaxis.set_major_formatter(FuncFormatter(_seconds_to_mmss))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=18)
    _format_axes(ax)
    fig.tight_layout()

    if len(results) > 1:
        plot_basename = f"combined_rescaled_predicted_vs_real_{base_name}_{style}"
    else:
        plot_basename = f"rescaled_{results[0]['label'].lower()}_predicted_vs_real_{base_name}_{style}"
    saved_paths = _save_figure(fig, output_folder, plot_basename)
    plt.close(fig)
    return saved_paths


def _simulation_result(label, models, forward_fn, X_seq, threshold, step_size, sequence_length):
    """Simulate one model on one file's windows and package it for plotting."""
    simulated_A, _probs, decisions = _simulate_voltage(models, forward_fn, X_seq, threshold, step_size)
    n_preds = len(simulated_A)
    print(f">>> [{label}] Ran {n_preds} predictions | step_size={step_size:.4f} | "
          f"predicted increases (prob>=t*): {int(decisions.sum())}/{n_preds}")
    return {
        "label": label,
        "sim_norm": _normalize_01(simulated_A),
        "sequence_length": sequence_length,
        "n_preds": n_preds,
    }


# ==========================================
# 5. Per-Run Evaluation
# ==========================================

def _load_run(checkpoint_path, label_prefix=""):
    """Load a checkpoint plus its t* and the settings needed for preprocessing and simulation."""
    models, forward_fn, scaler, checkpoint = load_checkpoint(checkpoint_path)
    threshold, threshold_source = _resolve_threshold(checkpoint, checkpoint_path)
    model_type = str(checkpoint.get("model_type", "cnn")).lower()
    _print_diagnostics(checkpoint, threshold, threshold_source,
                       label=f"{label_prefix}{model_type.upper()}")
    hp = checkpoint.get("hyperparameters", {}) or {}
    return {
        "models": models,
        "forward_fn": forward_fn,
        "scaler": scaler,
        "checkpoint": checkpoint,
        "threshold": threshold,
        "model_type": model_type,
        "label": model_type.upper(),
        "sequence_length": int(hp.get("sequence_length", 30)),
        "missing_value": float(hp.get("missing_value", 0.0)),
        "step_size": float(checkpoint.get("avg_step_size") or 1.0),
        "hyperparameters": hp,
    }


def _build_file_sequences(test_dfs, run):
    """Build windows per test file with this run's scaler and sequence length."""
    per_file = []
    for df in test_dfs:
        X_seq, y_seq = process_single_df_to_sequences(
            df, run["scaler"], run["sequence_length"], missing_value=run["missing_value"]
        )
        if len(X_seq) == 0:
            print(f"  SKIP {df.attrs.get('source_file')}: too short or no valid targets.")
            continue
        per_file.append((df, X_seq, y_seq))
    return per_file


def evaluate_run(run_name, checkpoint_path, test_dfs, test_files, run_output_folder,
                 compare_checkpoint_path=None):
    """Evaluate one checkpoint on the test files, save reports and voltage plots."""
    print("\n" + "#" * 80)
    print(f"RUN: {run_name}")
    print("#" * 80)
    os.makedirs(os.path.join(run_output_folder, "figures"), exist_ok=True)

    run = _load_run(checkpoint_path)
    per_file = _build_file_sequences(test_dfs, run)
    if not per_file:
        raise RuntimeError("No test sequences could be built from the test files.")

    X_test = torch.FloatTensor(np.concatenate([x for _, x, _ in per_file]))
    y_test = torch.FloatTensor(np.concatenate([y for _, _, y in per_file]))
    source_files = np.array(
        [df.attrs.get("source_file", "unknown_file") for df, x, _ in per_file for _ in range(len(x))],
        dtype=object,
    )
    pos_weight = torch.tensor(float(run["checkpoint"].get("pos_weight", 1.0)), dtype=torch.float)

    test_metrics = evaluate_test_set(
        run["models"], run["forward_fn"], X_test, y_test, source_files,
        pos_weight, threshold=run["threshold"],
    )
    save_test_metrics_report(run_output_folder, test_metrics, y_test, pos_weight,
                             checkpoint_path, test_files)

    other = None
    if compare_checkpoint_path is not None:
        other = _load_run(compare_checkpoint_path, label_prefix="COMPARISON ")
        if other["label"] == run["label"]:
            print(f"  NOTE: comparison checkpoint is also {other['label']}; skipping combined plot.")
            other = None

    figures_folder = os.path.join(run_output_folder, "figures")
    for df, X_seq, _ in per_file:
        results = [_simulation_result(run["label"], run["models"], run["forward_fn"], X_seq,
                                      run["threshold"], run["step_size"], run["sequence_length"])]
        for style in ("solid", "markers"):
            _plot_rescaled_voltage(df, results, figures_folder, style)

        if other is not None:
            X_other, _ = process_single_df_to_sequences(
                df, other["scaler"], other["sequence_length"], missing_value=other["missing_value"]
            )
            if len(X_other) == 0:
                continue
            results.append(_simulation_result(other["label"], other["models"], other["forward_fn"],
                                              X_other, other["threshold"], other["step_size"],
                                              other["sequence_length"]))
            results.sort(key=lambda r: r["label"])  # CNN first, then LSTM
            for style in ("solid", "markers"):
                _plot_rescaled_voltage(df, results, figures_folder, style)

    hp = run["hyperparameters"]
    return {
        "run_name": run_name,
        "run_folder": os.path.dirname(checkpoint_path),
        "model_type": run["model_type"],
        "weight_init_seed": hp.get("weight_init_seed"),
        "split_seed": hp.get("split_seed"),
        "output_folder": run_output_folder,
    }


# ==========================================
# 6. Main
# ==========================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate polgun_train.py checkpoints on a test CSV/folder at the "
                    "validation-chosen threshold and plot simulated vs real voltage."
    )
    p.add_argument("--checkpoint", default=None,
                   help="Mother folder, run folder, or full_checkpoint.pth")
    p.add_argument("--test", default=None,
                   help="Test CSV or folder of CSVs (default: test_dir stored in the checkpoint)")
    p.add_argument("--compare", default=None,
                   help="Optional checkpoint path of the other model type for the combined plot")
    return p.parse_args()


def main():
    args = _parse_args()

    checkpoint_input = args.checkpoint or _prompt_path(
        "Checkpoint path (mother folder, run folder, or full_checkpoint.pth): "
    )
    base_folder, runs = _discover_checkpoints(checkpoint_input)
    print(f"Found {len(runs)} checkpoint(s) under: {base_folder}")

    if args.test:
        test_input = os.path.abspath(args.test)
    else:
        first_ckpt = torch.load(runs[0][1], map_location="cpu", weights_only=False)
        default_test = first_ckpt.get("test_dir")
        test_input = _prompt_path(
            f"Test CSV or folder [Enter = {default_test}]: ", allow_empty=True
        ) or default_test
    if not test_input or not os.path.exists(test_input):
        raise FileNotFoundError(f"Test path not found: {test_input}")

    if args.compare:
        compare_input = os.path.abspath(args.compare)
    elif args.checkpoint:
        compare_input = None
    else:
        compare_input = _prompt_path(
            "Optional checkpoint path of the other model for the combined CNN + LSTM plot "
            "[Enter = skip]: ", allow_empty=True
        )
    compare_map = {}
    if compare_input:
        _, compare_runs = _discover_checkpoints(compare_input)
        compare_map = dict(compare_runs)
        if len(runs) == 1 and len(compare_runs) == 1:
            compare_map = {runs[0][0]: compare_runs[0][1]}

    test_files = _discover_test_csvs(test_input)
    test_name = os.path.splitext(os.path.basename(os.path.normpath(test_input)))[0]
    test_root_folder = os.path.join(base_folder, f"test_{test_name}")
    os.makedirs(test_root_folder, exist_ok=True)

    test_dfs = []
    for path in test_files:
        df = pd.read_csv(path)
        df.attrs["source_file"] = os.path.basename(path)
        df.attrs["source_csv"] = path
        test_dfs.append(df)

    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)
    print(f"Checkpoints : {len(runs)} under {base_folder}")
    print(f"Test files  : {[os.path.basename(p) for p in test_files]}")
    print(f"Comparison  : {compare_input or '(none)'}")
    print(f"Output      : {test_root_folder}")
    print("=" * 70)

    run_records = []
    for run_index, (run_name, ckpt_path) in enumerate(runs, start=1):
        record = evaluate_run(
            run_name, ckpt_path, test_dfs, test_files,
            run_output_folder=os.path.join(test_root_folder, run_name),
            compare_checkpoint_path=compare_map.get(run_name),
        )
        record["run_index"] = run_index
        run_records.append(record)

    aggregate_multi_seed_results(test_root_folder, run_records)
    print(f"\nAll test outputs saved to: {test_root_folder}")


if __name__ == "__main__":
    main()
