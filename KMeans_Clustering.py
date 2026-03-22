import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def apply_kmeans_clustering():
    print("1. Loading the database...")
    # FIXED: Reverted to the correct database name
    conn = sqlite3.connect('energy_db.sqlite')

    # Load the data we just created
    df = pd.read_sql_query("SELECT * FROM advanced_energy_data", conn)

    # FIXED: Corrected the typo 'Datatima' to 'Datetime'
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    print("2. Preparing weather features for K-Means...")
    # We cluster based purely on weather conditions to find "climate profiles"
    weather_features = ['temperature_c', 'humidity_percent', 'wind_speed']
    X_weather = df[weather_features].values

    # Scaling is critical for K-Means to ensure equal weight across all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weather)

    print("3. Running K-Means Clustering (K=4 weather profiles)...")
    # Define 4 distinct weather clusters (e.g., Extreme Heat, Freezing, Mild, Wet)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['weather_cluster'] = kmeans.fit_predict(X_scaled)

    print("4. Updating the SQLite database with the new 'weather_cluster' feature...")
    # Overwrite the table with the newly added feature column
    df.to_sql('advanced_energy_data', conn, if_exists='replace', index=False)

    # create the index to maintain fast query speeds
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datetime ON advanced_energy_data(Datetime);")
    conn.close()

    print("5. Generating a visualization of the Weather Clusters...")
    # Sample 10,000 random points so the scatter plot renders efficiently without freezing
    df_sample = df.sample(n=10000, random_state=42)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='temperature_c',
        y='humidity_percent',
        hue='weather_cluster',
        palette='viridis',
        data=df_sample,
        alpha=0.6
    )
    plt.title('Weather Profiles Discovered by K-Means (Temperature vs. Humidity)')
    plt.xlabel('Average Temperature (C)')
    plt.ylabel('Average Humidity (%)')
    plt.legend(title='Cluster ID')
    plt.grid(True)

    print("   Close the graph window to complete the execution.")
    plt.show()

    print("-" * 60)
    print("SUCCESS! Unsupervised learning complete.")
    print("New feature 'weather_cluster' successfully added to the database.")
    print("-" * 60)


if __name__ == "__main__":
    apply_kmeans_clustering()