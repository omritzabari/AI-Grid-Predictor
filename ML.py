import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


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


def find_optimal_k_for_knn(X, y, max_k=30):
    print(f"\n2. Finding optimal K for KNN (Testing 1 to {max_k})...")
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


def evaluate_models(X, y, optimal_k):
    print("\n3. Comparing Linear Regression vs. KNN on the FULL dataset (5-Fold CV)...")
    print("   [This will take a moment, please wait...]")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    lr_r2_scores, lr_rmse_scores = [], []
    knn_r2_scores, knn_rmse_scores = [], []

    fold_number = 1
    for train_index, test_index in kf.split(X):
        print(f"   Processing Fold {fold_number}/5...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Linear Regression (Using all PCA explanatory variables)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_r2_scores.append(r2_score(y_test, lr_preds))
        lr_rmse_scores.append(np.sqrt(mean_squared_error(y_test, lr_preds)))

        # KNN Regression
        knn = KNeighborsRegressor(n_neighbors=optimal_k, n_jobs=-1)
        knn.fit(X_train, y_train)
        knn_preds = knn.predict(X_test)
        knn_r2_scores.append(r2_score(y_test, knn_preds))
        knn_rmse_scores.append(np.sqrt(mean_squared_error(y_test, knn_preds)))

        fold_number += 1

    print("\n" + "=" * 60)
    print("FINAL OVERALL COMPARISON (Average of all 5 folds)")
    print("=" * 60)
    avg_lr_r2 = np.mean(lr_r2_scores) * 100
    avg_lr_rmse = np.mean(lr_rmse_scores)
    avg_knn_r2 = np.mean(knn_r2_scores) * 100
    avg_knn_rmse = np.mean(knn_rmse_scores)

    print("1. Multiple Linear Regression:")
    print(f"   Overall R-squared: {avg_lr_r2:.2f}%")
    print(f"   Overall RMSE:      {avg_lr_rmse:.2f} MW\n")
    print(f"2. K-Nearest Neighbors [KNN with K={optimal_k}]:")
    print(f"   Overall R-squared: {avg_knn_r2:.2f}%")
    print(f"   Overall RMSE:      {avg_knn_rmse:.2f} MW\n")

    if avg_lr_r2 > avg_knn_r2:
        print("OVERALL WINNER: Linear Regression (Surprising!)")
    else:
        print("OVERALL WINNER: K-Nearest Neighbors (KNN)")
    print("=" * 60)


def main():
    X, y = load_preprocessed_data()
    optimal_k = find_optimal_k_for_knn(X, y, max_k=30)
    evaluate_models(X, y, optimal_k)


if __name__ == "__main__":
    main()