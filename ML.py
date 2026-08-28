import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Longest look-back window used as a feature (lag_168h_consumption).
# Used as the gap between train and test so that features at the seam
# cannot reach back into the training period.
LAG_HOURS = 168


def load_preprocessed_data():
    print("1. Loading Massive PCA data from disk...")
    try:
        X_pca = joblib.load('pca_transformed_data.pkl')
        y = joblib.load('target_variable_y.pkl')
        print(f"   Data loaded successfully! Matrix shape: {X_pca.shape}")
        return X_pca, y
    except FileNotFoundError:
        print("   ERROR: Could not find the saved data files (.pkl).")
        exit()


def load_naive_baseline(n_rows):
    """
    Load the 24-hour lag column, which is the naive baseline: predict that
    consumption will be whatever it was at the same hour yesterday.

    This also verifies the assumption TimeSeriesSplit depends on. TimeSeriesSplit
    splits by ROW POSITION, not by date, so it is only meaningful if row order
    equals chronological order. Rather than assume that, check it and fail loudly.
    """
    print("\n2. Loading the naive baseline and verifying chronological order...")
    conn = sqlite3.connect('energy_db.sqlite')
    df = pd.read_sql_query(
        "SELECT Datetime, lag_24h_consumption FROM advanced_energy_data", conn)
    conn.close()

    if len(df) != n_rows:
        raise ValueError(
            f"Row mismatch: the database holds {len(df):,} rows but the PCA matrix "
            f"holds {n_rows:,}. Re-run PCA.py so the two stay in sync."
        )

    if df['lag_24h_consumption'].isna().any():
        raise ValueError(
            "lag_24h_consumption contains missing values, so the baseline cannot "
            "be scored. Re-run Build_DataBase.py."
        )

    timestamps = pd.to_datetime(df['Datetime'])
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Rows in 'advanced_energy_data' are not in chronological order. "
            "TimeSeriesSplit splits by row position, so the table must be sorted "
            "by Datetime for this evaluation to mean anything."
        )

    print(f"   OK: {len(df):,} rows, {timestamps.iloc[0]} -> {timestamps.iloc[-1]}")
    return df['lag_24h_consumption'].to_numpy()


def find_optimal_k_for_knn(X, y, max_k=30):
    print(f"\n3. Finding optimal K for KNN (Testing 1 to {max_k})...")
    print("   [Using a 25% random sample of the massive dataset for speed]")
    best_k = 1
    best_score = -float('inf')

    # Take a random 25% sample for the K-search speed
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=int(len(X) * 0.25), replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    # Test only ODD numbers for K (standard practice to prevent ties)
    for k in range(1, max_k + 1, 2):
        knn_temp = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        scores = cross_val_score(knn_temp, X_sample, y_sample, cv=3, scoring='r2')
        avg_score = scores.mean()

        print(f"   Testing K={k} -> R2 Score: {avg_score * 100:.2f}%")
        if avg_score > best_score:
            best_score = avg_score
            best_k = k

    print(f"   -> The Optimal K found is: {best_k}")

    # Save the optimal K to a text file so the production script can use it automatically
    with open('best_k.txt', 'w') as f:
        f.write(str(best_k))

    return best_k


def evaluate_models(X, y, optimal_k, naive):
    print("\n4. Comparing the naive baseline, Linear Regression and KNN...")
    print(f"   [TimeSeriesSplit, 5 folds, gap={LAG_HOURS}h - always train on the past,")
    print("    test on the future. Shuffling here would leak, because consecutive")
    print("    hours are nearly identical and KNN would simply retrieve the answer.]")

    tscv = TimeSeriesSplit(n_splits=5, gap=LAG_HOURS)

    labels = {
        'naive': 'Naive baseline (same hour yesterday)',
        'lr': 'Multiple Linear Regression',
        'knn': f'K-Nearest Neighbors (K={optimal_k})',
    }
    scores = {name: {'r2': [], 'rmse': []} for name in labels}

    def record(name, truth, prediction):
        scores[name]['r2'].append(r2_score(truth, prediction))
        scores[name]['rmse'].append(np.sqrt(mean_squared_error(truth, prediction)))

    for fold_number, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f"   Fold {fold_number}/5   train={len(train_index):,}   test={len(test_index):,}")

        # Naive baseline - no model at all, just yesterday's value at this hour.
        record('naive', y_test, naive[test_index])

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        record('lr', y_test, lr.predict(X_test))

        knn = KNeighborsRegressor(n_neighbors=optimal_k, n_jobs=-1)
        knn.fit(X_train, y_train)
        record('knn', y_test, knn.predict(X_test))

    order = ['naive', 'lr', 'knn']

    print("\n" + "=" * 72)
    print("FINAL COMPARISON  (mean over 5 chronological folds)")
    print("=" * 72)
    for name in order:
        mean_r2 = np.mean(scores[name]['r2']) * 100
        mean_rmse = np.mean(scores[name]['rmse'])
        print(f"{labels[name]:<38s}  R2 = {mean_r2:6.2f}%   RMSE = {mean_rmse:8.1f} MW")

    print("-" * 72)
    print("Per-fold R2 (%). The first fold trains on the least data, so a low")
    print("first value means the model is still data-hungry rather than broken:")
    for name in order:
        per_fold = "  ".join(f"{value * 100:5.1f}" for value in scores[name]['r2'])
        print(f"   {labels[name]:<38s} {per_fold}")
    print("=" * 72)


def main():
    X, y = load_preprocessed_data()
    naive = load_naive_baseline(len(X))
    optimal_k = find_optimal_k_for_knn(X, y, max_k=30)
    evaluate_models(X, y, optimal_k, naive)


if __name__ == "__main__":
    main()
