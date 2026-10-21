# AI-Based Time-Series Modeling for High-Voltage DC Photocathode Gun Conditioning

This repository contains the data-processing, analysis, training, and test-evaluation scripts for CNN and LSTM models that classify the next voltage-conditioning action of a high-voltage DC photocathode gun. The models are trained on historical diagnostic data.

## Associated publication

**AI-Based Time-Series Modeling for High-Voltage DC Photocathode Gun Conditioning**  
Authors: Sai Sumanth Kantamneni
Journal: Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment  
Status: Accepted 
DOI: 

Please cite the associated publication if you use this work. A BibTeX citation can be added once the bibliographic details are final.

## Data availability 
The experimental data supporting this study are available from the corresponding author upon request. Raw and processed experimental data are **not included** in this repository.


## Repository layout

```text
.
├── README.md
├── data_preparation/
│   ├── polgun_datacombine.py
│   └── polgun__cleaning.py
├── polgun_correlations.py
├── polgun_train.py
└── polgun_test.py
└── minirocket_LR_SGD.py

```

| Script | Purpose |
| --- | --- |
| `data_preparation/polgun_datacombine.py` | Raw sensor csv files are combined using time stamps. Each sensor data file needs to be combined manually |
| `data_preparation/polgun__cleaning.py` | Applies voltage cleaning procedure and labelling of data to be used by the polgun_train.py file |
| `polgun_correlations.py` | Optional lagged Pearson correlation analysis of current, pressure, and total radiation, aggregated across input files; saves figures and a CSV summary. |
| `polgun_train.py` | Contains both CNN and LSTM architectures, window construction, training/validation splitting, normalization, model fitting, validation threshold selection, and checkpoint/report generation. |
| `polgun_test.py` | Loads saved checkpoints, evaluates held-out CSV data, aggregates metrics, and plots predicted versus measured voltage traces after rescaling. |
| `minirocket_LR_SGD.py` | MiniROCKET + logistic regression (SGD) is the model used for comparison |


## Requirements

Required packages can be found in the requirements.txt file and can be installed by running the command:
```bash
python -m pip install -r requirements.txt
```


##Expected Inputs
The preparation scripts expect CSV files with a `Time` column and diagnostic fields. The cleaning script expects the following fields (missing fields are added as `NaN`, but downstream analyses may require them to contain valid data):

```text
Time
glassmanDataXfer:hvPsVoltageMeasM
gc1_diff
gc2_diff
gc3_diff
gc4_diff
GunCurrent.Avg
peg-BL-cc:pressureM
```

Cleaning additionally produces:

- `VoltageChange`: `1` for a positive change in voltage from the current row to the next, `0` for no change, `-1` for a negative change, and `NaN` where the difference is undefined.
- `RadiationTotal`: sum of the available `gc1_diff` through `gc4_diff` values, using `NaN` when all four are missing.

The training and test scripts use the columns `GunCurrent.Avg`, `peg-BL-cc:pressureM`, `RadiationTotal`, and `VoltageChange`. The measured voltage column is **not** a model input; it is used in preparation and for voltage-trajectory comparisons.

## Workflow

### 1. Combine diagnostic files (as needed)

Edit the file names, directory, selected column indices, and operation flags in the `if __name__ == "__main__"` block of `data_preparation/polgun_datacombine.py`.

The code will combine sensors by matching the timestamps. This needs to be done manually by including the sensor files that are needed for the model. For example: In line 201 and 202, the sensor file names needs to be included for them to be combined.

```bash
python data_preparation/polgun__datacombine.py
```

### 2. Clean, label, and truncate each conditioning run

Set `input_dir` and `output_dir` in the bottom of `data_preparation/polgun__cleaning.py`, then run:

```bash
python data_preparation/polgun__cleaning.py
```

For each CSV in the input folder, this script:

1. Removes any duplicate timestamps
2. Voltage cleaning and preparation is done as per the cleaning procedure outlined in the associated publication.
3. Writes only the expected diagnostic fields and the two generated labels.

Output files follow the pattern:

```text
processed_<original_name>_uptoMax_plus10.csv
```

### 3. Separate training and held-out test data

Place the prepared CSVs into separate training and test directories, then set `TRAIN_DIR` and `TEST_DIR` near the top of `polgun_train.py`.

**Important:** The script does not automatically create a held-out test split by experimental run. The training directory and the separate test directory must be chosen deliberately. `TEST_DIR` is recorded in checkpoints, but is not read during training.

### 4. Optional diagnostic correlation analysis

Set `TRAIN_DIR` in `analysis/polgun_correlations.py` and run:

```bash
python analysis/polgun_correlations.py
```

The script calculates linear correlation for current–pressure, current–radiation, and pressure–radiation over lags from -50 to +50 seconds (default), then computes mean and median curves across files. It reports peak-correlation lag, correlation at that peak, and full width at half maximum (FWHM). Results are saved in `analysis/crosscorr_figures/` when the file is kept in the suggested layout.

This step is not required for model training or testing.

### 5. Train the CNN or LSTM

In `polgun_train.py`, set:

```python
MODEL_TYPE = 'cnn'  # or 'lstm'
```

The training script's default configuration includes:

| Setting | Default |
| --- | --- |
| Sequence length | 30 samples |
| Learning rate | `0.000005` |
| Maximum epochs | 50 |
| Batch size | 128 |
| CNN convolution filters | 128, 128, 256 |
| CNN kernel size | 5 |
| CNN max-pool size | 2 |
| LSTM | 2 layers, hidden dimension 64 |
| Early stopping patience | 15 epochs |
| Threshold criterion | Maximum validation MCC |
| Weight-initialization seeds | 33, 545, 97, 123, 999 |
| Training/validation split seeds | 12, 97, 435, 8765, 88 |

**Note:** The single shared learning-rate setting defaults to `5e-6` even if `MODEL_TYPE` is switched to `lstm`. Change it explicitly if the intended LSTM experiment uses a different learning rate.

Run:

```bash
python polgun_train.py
```

By default, this launches **25 independent training processes**, one for each combination of five weight-initialization seeds and five split seeds. To run a single experiment instead, set the environment variable `POLGUN_SINGLE_RUN=1` before executing the script. The single-run defaults are weight-initialization seed `545` and split seed `97`; `POLGUN_WEIGHT_INIT_SEED` and `POLGUN_SPLIT_SEED` can override them.

For each run, the script:

1. Builds overlapping sliding windows with four channels: current, pressure, radiation, and previous action (`Prev_A1`). Each window predicts the action at the following time step.
2. Maps `VoltageChange=-1` to `0` and forces the next five time steps to `0` in the sequence-processing routine.
3. Randomly shuffles and splits the windows into 85% training and 15% validation.
4. Computes normalization statistics from training-window sensor values only; the previous-action channel is not standardized. Missing sensor readings are replaced with `0` after normalization.
5. Trains with Adam and class-weighted `BCEWithLogitsLoss`, where the positive-class weight is the ratio of training Hold to Increase examples.
6. Restores weights from the epoch with the lowest validation loss.
7. Selects a classification threshold from validation predictions, using maximum MCC by default.
8. Saves the model weights, full checkpoint, learning curves, validation metrics, and training-environment information.

A typical multi-seed output folder contains:

```text
polgun_cnn_multiseed_<timestamp>/
├── run_01_init_33_split_12/
│   ├── checkpoint.pth
│   ├── full_checkpoint.pth
│   ├── epoch_history.json
│   ├── model_architecture.json
│   ├── training_summary.txt
│   ├── val_threshold_selection.json
│   ├── val_pr_curve_points.csv
│   ├── system_and_timing.json
│   └── figures/
├── run_02_init_33_split_97/
│   └── ...
├── ...
├── all_runs_validation_metrics.csv
├── aggregate_validation_metrics_mean_sd.csv
└── aggregate_validation_metrics_mean_sd.json
```

`checkpoint.pth` contains model weights; `full_checkpoint.pth` additionally contains model type, hyperparameters, scaler statistics, training history, selected validation threshold, and data-directory information. The multi-seed summary reports means and sample standard deviations across completed runs.

### 6. Evaluation

Run with explicit paths:

```bash
python polgun_test.py --checkpoint "path/to/polgun_cnn_multiseed_<timestamp>" --test "path/to/test_csv_folder"
```

Or run interactively:

```bash
python polgun_test.py
```

The checkpoint argument can be a multi-seed parent directory, an individual run directory, or a `full_checkpoint.pth` file. The test argument can be one CSV or a folder of CSV files. Without explicit arguments, the script prompts for paths and can offer the test directory stored in the checkpoint.

It uses the threshold committed during validation and computes accuracy, precision, recall, F1, average precision (reported as PR-AUC), MCC, and a confusion matrix. It also saves per-file metrics and mean/SD summaries across evaluated checkpoints.

Example outputs:

```text
test_<test_name>/
├── run_01_init_33_split_12/
│   ├── test_metrics.json
│   ├── test_metrics_by_file.csv
│   └── figures/
├── ...
├── all_runs_test_metrics.csv
├── aggregate_test_metrics_mean_sd.csv
└── aggregate_test_metrics_mean_sd.json
```

For an optional combined CNN/LSTM comparison plot, supply a checkpoint from the other model type:

```bash
python polgun_test.py --checkpoint "path/to/cnn_runs" --test "path/to/test_csv_folder" --compare "path/to/lstm_runs"
```

For multi-seed comparisons, corresponding CNN and LSTM checkpoints are matched by run-folder name.


### 7. Baseline 

```bash
python minirocket_ridge.py
```

Trains the model used for comparison. It produces both MiniROCKET + Ridge classifier and MiniROCKET + Logistic Regression (SGD). The MiniROCKET + Logistic Regression (SGD) model was used for our comparison.


## Reproducibility and release notes

- Update the hard-coded local Windows paths in the supplied scripts before running them on another machine.
- Keep `polgun_train.py` and `polgun_test.py` in the same directory unless you update the import in the test script.
- Use the exact train/test file selection, cleaning settings, and model hyperparameters associated with the reported experiment.
- The 25 seed combinations are repeated computational runs, not 25 independent experimental datasets.
- The `full_checkpoint.pth` files store local training/test paths. Inspect checkpoints and output reports before sharing them publicly.

## Citation

If you use this repository, please cite the associated journal article:

> [Complete publication citation to be added when available.]

