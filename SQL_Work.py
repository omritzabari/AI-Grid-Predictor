import sqlite3
import pandas as pd


def apply_sql_feature_engineering():
    print("1. Connecting to the database...")

    # Create a connection object to our database file
    conn = sqlite3.connect('energy_db.sqlite')

    # Create a cursor to execute SQL commands
    cursor = conn.cursor()

    print("2. Ensuring INDEX exists for performance...")
    # Create index to drastically speed up queries on the massive dataset
    cursor.execute("""CREATE INDEX IF NOT EXISTS idx_datetime ON advanced_energy_data(Datetime);""")

    print("3. Creating an advanced VIEW with CASE WHEN for feature engineering...")
    # Create a virtual table (VIEW) that adds logical categories on the fly
    create_view_query = """ 
        CREATE VIEW IF NOT EXISTS ml_feature_view AS
        SELECT 
            *,
            -- Feature 1: Classify time into demand shifts (Peak vs. Off-Peak)
            CASE 
                WHEN CAST(hour AS INTEGER) BETWEEN 16 AND 21 THEN 'Peak_Demand'
                WHEN CAST(hour AS INTEGER) BETWEEN 0 AND 6 THEN 'Low_Demand'
                ELSE 'Normal_Demand'
            END AS demand_shift,

            -- Feature 2: Classify weather into categories based on temperature
            CASE 
                WHEN temperature_c < 10 THEN 'Cold'
                WHEN temperature_c BETWEEN 10 AND 25 THEN 'Pleasant'
                ELSE 'Hot'
            END AS weather_condition
        FROM advanced_energy_data;
        """

    cursor.execute(create_view_query)
    conn.commit()

    print("4. Fetching data from the new VIEW to check the result:")
    # Pull a few rows to verify the new columns were added correctly
    test_query = """
            SELECT Datetime, consumption, hour, demand_shift, weather_condition 
            FROM ml_feature_view 
            LIMIT 10;
        """
    df_features = pd.read_sql_query(test_query, conn)
    print(df_features.to_string(index=False))

    conn.close()
    print("-" * 60)
    print("SUCCESS! SQL feature engineering complete.")
    print("View 'ml_feature_view' created in the database.")
    print("-" * 60)


if __name__ == "__main__":
    apply_sql_feature_engineering()