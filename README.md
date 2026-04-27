# RELeASE Matrix Training

This project trains a proton-forecasting lookup matrix from historical SEP data and then uses that trained matrix for inference.

## What the notebook does

The notebook [train_delta_matrix.ipynb](train_delta_matrix.ipynb) trains delta-intensity matrices from historical CSV files in the `gsep_ts` archive, excluding `GSEP_List.csv`.

The training flow is:

1. Read every historical CSV from the `gsep_ts` folders.
2. Clean the electron flux with a rolling-median background subtraction.
3. Compute the training features for each timestamp:
   - current electron intensity
   - electron change over a configurable lookback window
4. Pair those features with the proton flux at the chosen lead time.
5. Bin the feature pairs into a 2D matrix.
6. Fill each matrix cell with the median proton target and a confidence score.
7. Save the result as one JSON matrix file or a matrix library (multiple windows).

## Unified model and matrix library

The framework is one model with consistent physics, but it can operate with a library of matrices:

- Instrument dimension: each instrument has its own calibration matrix set.
- Time-window dimension: each instrument can have multiple sliding-window calibrations, such as 30, 60, and 90 minutes.

This means one alert workflow can query multiple calibrated matrices while preserving a single forecasting logic.

## Train multiple matrices from CLI

Use [train_delta_matrix.py](train_delta_matrix.py) with an instrument label and one or more sliding windows.

Example (three windows for SOHO):

```bash
python train_delta_matrix.py /path/to/*.csv \
   --time-col time_tag \
   --electron-col p4 \
   --proton-col p3_flux_ic \
   --instrument SOHO \
   --sliding-window-minutes 30 60 90 \
   --lead-minutes 60 \
   --out-dir ./matrices
```

This writes files such as:

- `matrix_soho_30min.json`
- `matrix_soho_60min.json`
- `matrix_soho_90min.json`
- `matrix_index_soho.json` (index/library file)

Legacy single-window usage still works via `--delta-window-minutes` and `--out-json`.

## Run REleASE with instrument + window selection

Use [RELease.py](RELease.py) in `delta-matrix` mode with either:

- `--delta-matrix-json` for one matrix, or
- `--delta-matrix-library-json` for indexed multi-matrix selection.

Example (library mode):

```bash
python RELease.py \
   --model delta-matrix \
   --instrument SOHO \
   --delta-window-minutes 60 \
   --delta-matrix-library-json ./matrices/matrix_index_soho.json \
   --electron-csv /path/electron.csv \
   --electron-time-col time_tag \
   --electron-flux-col p4 \
   --proton-csv /path/proton.csv \
   --proton-time-col time_tag \
   --proton-gt10-col p3_flux_ic \
   --out-json ./release_output.json
```

If the exact requested window is not present for the selected instrument, RELeASE uses the nearest available calibrated window for that instrument.

## Why `GSEP_List.csv` is excluded

`GSEP_List.csv` is the event list used for checking accuracy. It is not used as calibration input because it stores the real SEP events we want to compare against later.

## How the matrix is used

The trained matrix JSON is loaded by [RELease.py](RELease.py) in delta-matrix mode. At inference time, the model:

- reads the latest electron flux,
- computes the current intensity and delta feature,
- looks up the matching matrix cell,
- predicts the proton flux at the configured lead time.

## Running the notebook

Edit the notebook parameters if needed, then run the cells top to bottom. The notebook will:

- train the matrix from all historical CSVs,
- write a single matrix JSON file,
- run the RELease model using that trained matrix.

## Data location

Historical data is stored under:

- `/Users/ran/cs/oires/data/raw/gsep_ts/`

The notebook is configured to use all `*.csv` files under that tree except `GSEP_List.csv`.