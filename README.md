# PredictMTA

![PredictMTA logo](logo.png)

PredictMTA is a Python application that estimates the probability of subway disruptions at individual New York City subway stops. It includes a Tkinter dashboard for exploring model predictions by line, direction, stop, and hour.

## Features

- Displays the model-estimated disruption probability for each line, direction, stop, and hour.
- Shows the next-hour outlook by default, limited to stops with a predicted probability of 50% or higher.
- Provides an explorer with filters for subway line, direction, stop, and hour, plus sortable, paginated results.
- Categorizes predicted probability as **Low** (under 25%), **Moderate** (25% to under 50%), or **High** (50% or higher).
- Includes light and dark display modes.
- Supports running predictions for a selected date and hour.

The displayed values are the model's estimated probabilities for a disruption under its training definition, not guarantees that incidents will occur. The model uses `predict_proba`; its probability outputs are not explicitly calibrated.

## Requirements

- Windows, macOS, or Linux with Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- Tkinter for the desktop interface. It is included with many Python installations; some Linux distributions provide it as a separate system package.
- Internet access to download the latest source data used for predictions.

Install the project dependencies from the repository root:

```shell
uv sync
```

## Included model

The repository includes [`models/model.joblib`](models/model.joblib), a trained model built from approximately one year of subway data. You can run the dashboard with this model right away; rebuilding the dataset and retraining are **not required** unless you want to update or experiment with the model.

## Run the dashboard

Start the dashboard for the current prediction date. The selected hour follows the next hour in New York time:

```shell
uv run python main.py
```

Pass a date to run predictions for that date. The dashboard defaults to midnight:

```shell
uv run python main.py 2026-10-31
```

Optionally, provide the hour as a 24-hour integer from `0` (12 AM) through `23` (11 PM):

```shell
uv run python main.py 2026-10-31 5
```

This selects 5 AM on October 31, 2026. The hour argument requires a date. Dates must use `YYYY-MM-DD`.

The explorer's time filter starts at the selected target hour. You can choose another hour or **All hours**. The next-hour outlook displays only stops with a high predicted probability (50% or higher); if none qualify, it says there are no stops above that probability threshold for the selected hour.

## Optional: rebuild the dataset and retrain

If you want to retrain the included model, build a dataset and train a replacement:

1. In `train/process.py`, adjust the date range in its `__main__` block if needed.
2. Download and process the configured days into a dataset:

   ```shell
   uv run python train/process.py
   ```

   Daily source files are downloaded from [subwaydata.nyc](https://subwaydata.nyc/) and processed into cached Parquet bins under `data/interim/`. The combined dataset is written to `data/dataset.parquet`. This step can take a while and requires internet access.

3. Train and save the model:

   ```shell
   uv run python train/train.py
   ```

   The trained model and its normal-headway reference data are saved to `models/model.joblib`.

Training saves the updated model to `models/model.joblib`, replacing the existing model. Start the dashboard using the commands above.

The prediction script downloads data for a source day and uses it to create predictions for 24 hours at each modeled stop. In current-date mode, the source day is calculated as 36 hours before the current time in New York. When a target date is supplied, the previous calendar day is used. Because source-data publication can lag, the requested data day must be available from subwaydata.nyc.

## Project layout

```text
main.py             Tkinter dashboard and command-line arguments
logo.png            Branding displayed in the dashboard and window icon
predict.py          Loads the model and generates stop-level predictions
train/
  process.py        Downloads and processes daily subway data
  train.py          Builds features, trains, evaluates, and saves the model
data/
  raw/              Downloaded source data
  interim/          Cached daily Parquet bins
  dataset.parquet   Combined training dataset
models/
  model.joblib      Trained model and normal-headway reference data
stops.txt           Stop ID-to-name lookup used by predictions
routes.txt          Route lookup data
```

Downloaded source files and generated datasets are local runtime artifacts. The included model is available at `models/model.joblib`; the dashboard does not need the training dataset unless you choose to retrain.
