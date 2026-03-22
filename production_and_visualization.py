import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
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
    # Split data (80/20) just for the visualization plot
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train the WINNING model
    model = KNeighborsRegressor(n_neighbors=optimal_k, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    # Create the visualization
    plt.figure(figsize=(10, 6))

    # Plot using alpha=0.3 to handle high density of points elegantly
    plt.scatter(y_test, y_pred, color='blue', alpha=0.3, label='Predicted vs Actual')

    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2,
             label='Perfect Prediction ($y=x$)')

    plt.title(f'10-Year Electricity Consumption: Actual vs. Predicted (KNN R-squared: {score * 100:.2f}%)', fontsize=14)
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