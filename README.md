# AI Grid Load Predictor

An end-to-end pipeline for hourly electricity demand forecasting on the PJM East grid, from raw data ingestion through to an interactive dashboard.

Ten years of hourly consumption data (2008–2017) are merged with weather observations from the Philadelphia airport station, stored in SQLite, enriched with time and lag features, and used to train and compare regression models. A Streamlit dashboard exposes the trained model for interactive forecasting and anomaly inspection.

This is a personal project, built to take methods I had studied and run them end to end on real, messy data.

**Demo video:** [assets/demoGif.mp4](./assets/demoGif.mp4) — GitHub does not play embedded video inside a README, so the file has to be opened directly.

---

## Results

Evaluated with `TimeSeriesSplit` over 5 chronological folds — always training on the past and testing on the future — with a 168-hour gap between train and test. 87,330 hourly records.

| Model | R² | RMSE (MW) |
|---|---|---|
| Naive baseline (same hour yesterday) | 78.62% | 2,970.5 |
| Multiple Linear Regression | 80.36% | 2,847.9 |
| KNN Regressor (K=7) | **87.44%** | **2,281.6** |

Per-fold R², oldest fold first:

| Model | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| Naive baseline | 79.1 | 80.5 | 75.7 | 77.8 | 80.0 |
| Linear Regression | 82.1 | 82.3 | 78.4 | 79.2 | 79.9 |
| KNN (K=7) | 86.9 | 87.1 | 87.1 | 88.5 | 87.7 |

**Reading these numbers:**

- The naive baseline reaches 78.6% on its own. Any model that does not clearly beat that is not earning its complexity, so it is reported first rather than left out.
- Linear regression adds only 1.7 points over the baseline. That is a result in itself: the relationship is not linear. Consumption rises in extreme heat *and* in extreme cold, so a single linear coefficient on temperature cannot capture it.
- KNN adds 8.8 points over the baseline, and its per-fold spread (86.9–88.5) is *narrower* than the baseline's (75.7–80.5) — it is not only more accurate but more stable across periods.
- The KNN figure is still mildly optimistic: K was selected using shuffled cross-validation, and the scaler and PCA were fitted on the full dataset. See the open limitations below. The gap over the baseline is too large to be explained by that alone, but the number should be read as an upper estimate.

---

## Quickstart

```bash
git clone https://github.com/omritzabari/AI-Grid-Predictor.git
cd AI-Grid-Predictor

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The SQLite database and the trained model files are **not** committed to the repository — they are build artifacts. The pipeline below regenerates them from `PJME_hourly.csv` (which is committed) plus a live weather download.

### Run the pipeline in this order

The order matters: each script consumes what the previous one produced.

| Step | Command | What it does | Produces |
|---|---|---|---|
| 1 | `python Build_DataBase.py` | Loads the consumption CSV, downloads 10 years of hourly weather from Meteostat, merges them, builds time and lag features | `energy_db.sqlite` |
| 2 | `python SQL_Work.py` | Adds an index on `Datetime` and creates the `ml_feature_view` view | view + index |
| 3 | `python KMeans_Clustering.py` | Clusters the weather variables into climate profiles and writes `weather_cluster` back into the table | updated table |
| 4 | `python PCA.py` | Standardises the features and applies PCA retaining 90% of the variance | `scaler_model.pkl`, `pca_model.pkl`, `pca_transformed_data.pkl`, `target_variable_y.pkl` |
| 5 | `python ML.py` | Compares the naive baseline, Linear Regression and KNN with `TimeSeriesSplit`, and selects K | `best_k.txt` |
| 6 | `python production_and_visualization.py` | Trains the final KNN model and plots actual vs predicted | `final_knn_model.pkl` |
| 7 | `streamlit run app.py` | Launches the dashboard | — |

`python Statistics.py` can be run any time after step 1; it only reads from the database and prints results.

**Step 1 requires an internet connection** — the weather data is fetched from the Meteostat API at runtime, one year at a time. It takes a few minutes.

**Step 3 must run before step 4.** K-Means adds the `weather_cluster` column to the table that PCA reads; running PCA first silently produces a feature set without it.

---

## Pipeline

### 1. Data ingestion — `Build_DataBase.py`

Merges the PJM East hourly consumption series with hourly temperature, humidity and wind speed from Meteostat station 72408 (Philadelphia International Airport), joined on the timestamp. Adds calendar features (hour, day of week, month, day of year, weekend flag, season) and two lag features: consumption 24 hours earlier and 168 hours earlier. Written to SQLite as `advanced_energy_data`.

### 2. SQL layer — `SQL_Work.py`

Creates an index on `Datetime`, and a view `ml_feature_view` that uses `CASE WHEN` to bucket hours into peak / off-peak / normal shifts and temperatures into cold / pleasant / hot bands.

### 3. Clustering — `KMeans_Clustering.py`

Runs K-Means (K=4) on the weather columns alone to find recurring climate profiles, and writes the assigned cluster back into the table as an additional feature.

### 4. Statistics — `Statistics.py`

Welch's t-test comparing weekday against weekend consumption, a 95% confidence interval for the mean, and Z-score based outlier detection with a threshold of 3.

### 5. Dimensionality reduction — `PCA.py`

`StandardScaler` followed by PCA with `n_components=0.90`, so the number of components is chosen automatically to retain 90% of the variance. The fitted scaler and PCA objects are serialised so the dashboard can apply the identical transform to new user input.

### 6. Modelling — `ML.py`, `production_and_visualization.py`

A naive baseline, Linear Regression and a KNN regressor are compared under `TimeSeriesSplit`, reporting R-squared and RMSE per fold. The final model is then trained and serialised.

### 7. Dashboard — `app.py`

Streamlit + Plotly. The user picks a target date and weather conditions, and the app applies the saved scaler and PCA transform before calling the model. It also shows a 7-day historical trajectory and a live statistics panel querying SQLite directly.

---

## Fixed after review

I went back over this project some time after building it. These two problems were serious enough to change what the results mean, so they were fixed rather than only documented.

**Evaluation split.** The original version evaluated with `KFold(shuffle=True)`. On a time series this leaks: shuffling puts hour *t-1* into training and hour *t* into test, and since consecutive hours are nearly identical, KNN effectively retrieves the answer from its training set instead of predicting it. Replaced with `TimeSeriesSplit` — train on the past, test on the future — with a 168-hour gap between the two, so that lag features at the seam cannot reach back into the training period. `ML.py` now also verifies that rows are in chronological order before splitting, and fails loudly if they are not, since `TimeSeriesSplit` splits by row position rather than by date.

**No baseline.** The original version compared Linear Regression against KNN, with nothing to anchor either of them. Predicting "whatever consumption was at this hour yesterday" already reaches 78.6% R², which means a model scoring in the eighties is not automatically impressive. The baseline is now measured inside the same folds as the models, so the comparison is like for like.

---

## Known limitations

These are still open. They are listed rather than quietly ignored, because knowing where the weaknesses are matters more than pretending they are absent.

**Evaluation**

- **The scaler and PCA are fitted on the entire dataset before cross-validation splits it,** so the preprocessing statistics see the test folds. The fix is to wrap scaling, PCA and the model in a `Pipeline` so they are refitted inside each fold. The numerical effect here is probably small — a mean computed over 87k rows versus 70k is nearly identical — but the structure is wrong.
- **K was selected with shuffled cross-validation on a random 25% sample,** and on the same data later used to report the final score. Both bias the reported KNN figure optimistically. A final year held out and untouched during development would fix this.

**Features**

- **`shift(24)` shifts by 24 rows, not by 24 hours.** It is only correct while the series has no gaps. A lag should be computed by subtracting a duration from the timestamp and looking up that timestamp.
- **Categorical variables (`season`, `month`, `weather_cluster`) are fed to the model as integers,** which imposes a false ordering and a false notion of distance — it implies that December is further from January than from November. They should be one-hot encoded.

**Clustering and statistics**

- **K=4 in K-Means was not justified.** No elbow plot or silhouette analysis was run to support that choice.
- **The t-test assumes independent observations, and hourly consumption is strongly autocorrelated.** The resulting p-value is therefore not calibrated, even though the direction of the effect is real.

**Dashboard**

- **The "uncertainty range" shown in the dashboard is the standard deviation of the neighbouring points, not a calibrated prediction interval.** It should not be read as a confidence band.
- **The cluster labels in the UI are hardcoded,** while K-Means assigns cluster numbers arbitrarily on each run. A re-run can therefore relabel the clusters without anything visibly breaking.
- **The order of the features is an implicit contract between the training script and the dashboard,** with nothing declaring or validating it. Adding a feature in one place and not the other would fail silently.

---

## Technology

Python 3.13 · pandas · NumPy · scikit-learn (K-Means, PCA, KNN, Linear Regression, StandardScaler) · SciPy · SQLite · Streamlit · Plotly · Matplotlib · Seaborn · Meteostat API
