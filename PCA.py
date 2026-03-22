import sqlite3
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_data_for_pca():
    print("1. Connecting to the massive database...")
    conn = sqlite3.connect('energy_db.sqlite')
    # Select all columns. We will separate target (consumption) from features in the next step
    query = "SELECT * FROM advanced_energy_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Drop Datetime as it cannot be fed into PCA directly
    df.drop(columns=['Datetime'], inplace=True)
    return df


def standardize_features(df):
    print("2. Standardizing the feature set...")
    # Target variable is 'consumption', everything else is a feature for PCA
    target = df['consumption'].values
    # Create a list of all feature column names
    feature_columns = [col for col in df.columns if col != 'consumption']
    x = df[feature_columns].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    print("   --- Standardization Complete ---")
    print(f"   Original shape of X matrix: {x.shape}")

    # We MUST save the scaler to use it later in the Streamlit UI!
    joblib.dump(scaler, 'scaler_model.pkl')
    print("   Scaler saved as 'scaler_model.pkl' (Crucial for UI later).")

    return x_scaled, target, feature_columns


def apply_pca(x_scaled):
    print("\n3. Performing Dimensionality Reduction (PCA)...")
    # Dynamically reduce dimensions by forcing the algorithm to retain 90% of the variance
    pca = PCA(n_components=0.90)
    principal_components = pca.fit_transform(x_scaled)
    print(f"   Algorithm automatically chose {pca.n_components_} components to retain >= 90% variance.")
    print(f"   Original data shape: {x_scaled.shape}")
    print(f"   Reduced data shape: {principal_components.shape}")

    return pca, principal_components

def visualize_pca(pca_model):
    print("\n4. Generating PCA Cumulative Variance Graph...")
    variance_ratio_full = pca_model.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(variance_ratio_full) + 1), np.cumsum(variance_ratio_full), marker='o', linestyle='--',
             color='b')
    plt.title('PCA - Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.grid(True)

    print("   Close the graph window to finish execution and save files...")
    plt.show()


def main_pca():
    # Step 1: Get data
    df = get_data_for_pca()

    # Step 2: Standardize X and extract y
    x_scaled, y, feature_cols = standardize_features(df)

    # Step 3: Apply PCA
    pca_model, pc_data = apply_pca(x_scaled)

    # Step 4: Visualize
    visualize_pca(pca_model)

    # Step 5: Save everything to disk exactly like the old project
    print("\n5. Saving transformed data and models to disk...")
    # Save the matrices for the ML.py script
    joblib.dump(pc_data, 'pca_transformed_data.pkl')
    joblib.dump(y, 'target_variable_y.pkl')

    # Save the trained PCA model itself so we can transform new user inputs in the UI
    joblib.dump(pca_model, 'pca_model.pkl')

    print("-" * 60)
    print(
        "SUCCESS! PCA complete. Files saved: 'pca_transformed_data.pkl', 'target_variable_y.pkl', 'pca_model.pkl', 'scaler_model.pkl'.")
    print("-" * 60)


if __name__ == "__main__":
    main_pca()