\# AI Grid Load Predictor: Advanced Energy Demand Forecasting



\## Project Overview

This project presents a complete, end-to-end Data Engineering and Machine Learning pipeline designed for grid load prediction. The system ingests historical electricity consumption telemetry and integrates it with weather data to train a highly accurate forecasting model. Utilizing advanced feature engineering, unsupervised learning (clustering), dimensionality reduction (PCA), and statistical inference, the project culminates in a real-time, interactive dashboard for load monitoring, prediction, and anomaly detection.



\## Architecture and Pipeline



The project is highly modular, with each stage representing a crucial step in the data lifecycle:



\### 1. Data Ingestion and Database Construction (Build\_DataBase.py)

\* Data Integration: Combines 10 years (2008-2018) of historical power consumption data (PJME) with meteorological telemetry (temperature, humidity, wind speed) fetched dynamically via the Meteostat API.

\* Feature Engineering: Extracts temporal dynamics (hour, day of week, month, season, weekend binary) and generates critical historical lag features (T-24h and T-168h).

\* Storage: Persists the processed, merged dataset into a robust SQLite relational database (energy\_db.sqlite).



\### 2. SQL Data Engineering and Optimization (SQL\_Work.py)

\* Performance Indexing: Implements database indexing on the Datetime column to drastically reduce query latency on the massive dataset.

\* Advanced Views: Constructs a dynamic virtual table (ml\_feature\_view) utilizing SQL CASE WHEN statements to classify operational shifts (Peak/Off-Peak/Normal) and categorize weather conditions (Cold/Pleasant/Hot) directly at the database level.



\### 3. Statistical Inference and Anomaly Detection (Statistics.py)

\* Hypothesis Testing (T-Test): Conducts Welch's T-Test to mathematically analyze and prove a statistically significant variance in power consumption between weekdays and weekends.

\* Population Parameters: Calculates the 95% Confidence Interval for the true historical mean of the grid's power consumption.

\* Outlier Detection: Deploys Z-Score methodology (|Z| > 3.0) to identify, isolate, and log extreme historical grid events and statistical anomalies.



\### 4. Unsupervised Learning and Climate Profiling (KMeans\_Clustering.py)

\* K-Means Clustering: Applies the K-Means algorithm (K=4) exclusively to weather variables to discover latent climate profiles without human labeling.

\* Database Enrichment: Injects the resulting weather\_cluster metric back into the SQLite database as a new feature, enhancing the predictive model's contextual understanding.



\### 5. Dimensionality Reduction (PCA.py)

\* Data Standardization: Normalizes the feature space using StandardScaler to ensure uniform algorithmic weighting.

\* Principal Component Analysis (PCA): Compresses the high-dimensional feature set while retaining strictly >= 90% of the cumulative variance. This optimizes the computational efficiency and speed of the downstream algorithms.

\* Artifact Serialization: Exports transformers as .pkl files for seamless deployment in the production environment.



\### 6. Predictive Modeling and Evaluation (ML.py \& production\_and\_visualization.py)

\* Algorithm Evaluation: Compares Multiple Linear Regression against K-Nearest Neighbors (KNN) Regressor using rigorous 5-Fold Cross-Validation.

\* Hyperparameter Tuning: Programmatically searches and identifies the optimal K value for the KNN algorithm based on a data subset.

\* Production Build: Trains the final model on 100% of the dataset, visualizes the Actual vs. Predicted accuracy, and serializes the trained model as final\_knn\_model.pkl.



\### 7. Interactive Production Dashboard (app.py)

\* User Interface: A state-of-the-art web application built with Streamlit.

\* Forecasting: Allows users to input target dates and simulated weather metrics to generate real-time grid load forecasts.

\* Visualizations: Features advanced Plotly charts, including demand severity gauges and 7-day historical trajectory lines.

\* Analytics Integration: Integrates a live statistical analytics section directly querying the underlying SQLite database.



\## Technology Stack

\* Language: Python 3

\* Data Engineering \& Database: SQLite3, SQL, Pandas, Numpy

\* Machine Learning: Scikit-Learn (K-Means, PCA, KNN, Linear Regression, StandardScaler)

\* Statistics: SciPy (Welch's T-Test, Z-Score, Confidence Intervals)

\* Visualization: Streamlit, Plotly, Seaborn, Matplotlib

\* APIs: Meteostat

