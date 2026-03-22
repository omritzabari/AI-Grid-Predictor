import sqlite3
import pandas as pd
import scipy.stats as stats
import numpy as np


def get_data_from_db():
    print("1. Loading massive dataset from the database for statistical analysis...")
    conn = sqlite3.connect('energy_db.sqlite')
    # Fetching the prepared advanced dataset
    query = "SELECT * FROM advanced_energy_data"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def t_test_weekend_vs_weekday(df):
    print("\n2. Performing T-Test (Weekend vs. Weekday Consumption)...")
    # Split the dataset using the binary feature we engineered earlier
    weekend_data = df[df['is_weekend'] == 1]['consumption']
    weekday_data = df[df['is_weekend'] == 0]['consumption']

    # Perform an independent T-Test assuming unequal variances (Welch's t-test)
    t_stat, p_val = stats.ttest_ind(weekday_data, weekend_data, equal_var=False)

    print("-" * 60)
    print(f"Average Weekday Consumption: {weekday_data.mean():.2f} MW")
    print(f"Average Weekend Consumption: {weekend_data.mean():.2f} MW")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.4e}")

    if p_val < 0.05:
        print(
            "Conclusion: There is a statistically significant difference in consumption between weekends and weekdays.")
    else:
        print("Conclusion: No significant difference found.")
    print("-" * 60)


def confidence_interval(df):
    print("\n3. Calculating 95% Confidence Interval for Mean Consumption...")
    data = df['consumption'].dropna()
    mean_val = np.mean(data)

    # Calculate the standard error of the mean
    std_error = stats.sem(data)

    # Calculate the 95% confidence interval
    ci = stats.t.interval(0.95, df=len(data) - 1, loc=mean_val, scale=std_error)

    print("-" * 60)
    print(f"Sample Mean: {mean_val:.2f} MW")
    print(f"We are 95% confident that the true population mean of power consumption")
    print(f"lies strictly between {ci[0]:.2f} MW and {ci[1]:.2f} MW.")
    print("-" * 60)


def detect_outliers(df):
    print("\n4. Detecting Statistical Anomalies (Outliers) using Z-Score...")
    # Calculate absolute Z-scores to find how many standard deviations away each point is
    df['z_score'] = np.abs(stats.zscore(df['consumption']))

    # A Z-score greater than 3 is universally considered an outlier in a normal distribution
    outliers = df[df['z_score'] > 3.0]

    print("-" * 60)
    print(f"Total anomalies detected (|Z-Score| > 3): {len(outliers)} hours out of {len(df)} total hours.")

    if len(outliers) > 0:
        print("\nTop 5 Most Extreme Consumption Hours in the last 10 years:")
        # Sort to show the absolute most extreme events
        top_outliers = outliers.sort_values(by='z_score', ascending=False).head(5)
        cols_to_show = ['Datetime', 'consumption', 'temperature_c', 'season', 'z_score']
        print(top_outliers[cols_to_show].to_string(index=False))
    print("-" * 60)


def main_stats():
    df = get_data_from_db()
    t_test_weekend_vs_weekday(df)
    confidence_interval(df)
    detect_outliers(df)


if __name__ == "__main__":
    main_stats()