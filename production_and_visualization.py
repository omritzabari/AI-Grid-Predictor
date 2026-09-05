import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import os


def load_preprocessed_data():
    print("1. Loading massive PCA data from disk...")
    try:
        X_pca = joblib.load('pca_transformed_data.pkl')
        y = joblib.load('target_variable_y.pkl')
        return X_pca, y
    except FileNotFoundError:
        print("   ERROR: Saved data not found. Run PCA script first.")
        exit()


def get_optimal_k():
    print("2. Retrieving the Optimal K found during Machine Learning testing...")
    if os.path.exists('best_k.txt'):
        with open('best_k.txt', 'r') as f:
            best_k = int(f.read().strip())
        print(f"   -> Successfully loaded Optimal K = {best_k}")
        return best_k
    else:
        print("   -> Warning: 'best_k.txt' not found. Defaulting to K=15.")
        return 15


def visualize_model_accuracy(X, y, optimal_k):
    print(f"\n3. Generating Actual vs. Predicted scatter plot using KNN (K={optimal_k})...")
    # Chronological 80/20 hold-out: this is a time series, so a shuffled split
    # would let the model see future hours while predicting the past and leak.
    split_index = int(len(X) * 0.80)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train the WINNING model
    model = KNeighborsRegressor(n_neighbors=optimal_k, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Create the visualization
    plt.figure(figsize=(10, 6))

    # Plot using alpha=0.3 to handle high density of points elegantly
    plt.scatter(y_test, y_pred, color='blue', alpha=0.3, label='Predicted vs Actual')

    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2,
             label='Perfect Prediction ($y=x$)')

    plt.title('Illustrative Actual vs. Predicted (KNN) on the Held-Out Final Period\n'
              "Not the model's accuracy - the headline metric comes from ML.py's TimeSeriesSplit evaluation",
              fontsize=12)
    plt.xlabel('Actual Consumption (MW)', fontsize=12)
    plt.ylabel('Predicted Consumption (MW)', fontsize=12)
    plt.legend()
    plt.grid(True)

    print("   Close the graph window to continue to the final production step...")
    plt.show()


def train_and_save_final_model(X, y, optimal_k):
    print("\n4. Training the FINAL production model on 100% of the massive dataset...")

    # Initialize the final model using the exact dynamic optimal K
    final_model = KNeighborsRegressor(n_neighbors=optimal_k, n_jobs=-1)

    # Train it on ALL available data
    final_model.fit(X, y)

    print("5. Saving the final trained model to disk...")
    joblib.dump(final_model, 'final_knn_model.pkl')

    print("-" * 60)
    print(f"SUCCESS! The system has successfully learned 10 years of history with K={optimal_k}.")
    print("The final 'brain' is saved as 'final_knn_model.pkl'.")
    print("WE ARE READY TO BUILD THE WEB APPLICATION UI!")
    print("-" * 60)


def main():
    X, y = load_preprocessed_data()
    optimal_k = get_optimal_k()

    visualize_model_accuracy(X, y, optimal_k)
    train_and_save_final_model(X, y, optimal_k)


if __name__ == "__main__":
    main()