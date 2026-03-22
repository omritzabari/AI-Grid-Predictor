import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import scipy.stats as stats

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Grid Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATE MANAGEMENT ---
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}


# --- CACHED DATA & MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler_model.pkl')
        pca = joblib.load('pca_model.pkl')
        knn = joblib.load('final_knn_model.pkl')
        return scaler, pca, knn
    except Exception as e:
        st.error("System Error: ML models failed to load. Please verify the presence of .pkl files in the directory.")
        st.stop()


scaler_model, pca_model, knn_model = load_models()


@st.cache_data
def get_full_stats_data():
    """Fetches full historical data specifically for the statistics section."""
    conn = sqlite3.connect('energy_db.sqlite')
    query = "SELECT Datetime, consumption, temperature_c, humidity_percent, season, hour, is_weekend FROM advanced_energy_data"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    df['season_name'] = df['season'].map(season_map)
    return df


# --- DATABASE HELPER FUNCTIONS ---
def get_reference_datetime(target_dt):
    """Maps future dates back to our dataset's timeline (<= 2018)."""
    if target_dt.year > 2018:
        try:
            return target_dt.replace(year=2017)
        except ValueError:
            return target_dt.replace(year=2017, day=28)
    return target_dt


def get_historical_lag(ref_datetime, hours_back):
    past_time = ref_datetime - timedelta(hours=hours_back)
    conn = sqlite3.connect('energy_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT consumption FROM advanced_energy_data WHERE Datetime = ?",
                   (past_time.strftime('%Y-%m-%d %H:%M:%S'),))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 30000.0


def get_closest_weather_cluster(temp, hum, wind):
    conn = sqlite3.connect('energy_db.sqlite')
    query = f"""
        SELECT weather_cluster FROM advanced_energy_data 
        ORDER BY (ABS(temperature_c - {temp}) + ABS(humidity_percent - {hum}) + ABS(wind_speed - {wind})) ASC LIMIT 1
    """
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0


def fetch_historical_chart_data(ref_date, days=7):
    start_date = ref_date - timedelta(days=days)
    conn = sqlite3.connect('energy_db.sqlite')
    query = "SELECT Datetime, consumption FROM advanced_energy_data WHERE Datetime BETWEEN ? AND ? ORDER BY Datetime"
    df = pd.read_sql_query(query, conn,
                           params=(start_date.strftime('%Y-%m-%d %H:%M:%S'), ref_date.strftime('%Y-%m-%d %H:%M:%S')))
    conn.close()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df


# --- CLUSTER NAMING DICTIONARY ---
WEATHER_PROFILES = {
    0: "Optimal & Mild",
    1: "Hot & Humid (Summer Peak)",
    2: "Cold & Harsh (Winter Peak)",
    3: "Transitional & Breezy"
}


# --- REUSABLE INPUT FORM ---
def render_input_form(key_prefix):
    st.subheader("⏱️ Temporal Alignment")

    use_live_time = st.toggle("🕒 Use Live System Time", value=True, key=f"{key_prefix}_time_toggle")

    if use_live_time:
        now = datetime.now()
        target_dt = now
        st.info(f"Targeting live conditions: {now.strftime('%b %d, %Y - %H:%M')}")
    else:
        col_d, col_t = st.columns(2)
        with col_d:
            selected_date = st.date_input("Target Date", value=datetime(2026, 3, 22), key=f"{key_prefix}_date")
        with col_t:
            selected_time = st.time_input("Target Time", value=datetime.strptime("14:00", "%H:%M").time(),
                                          key=f"{key_prefix}_time")
        target_dt = datetime.combine(selected_date, selected_time)

    st.subheader("🌤️ Meteorological Parameters")
    temp = st.slider("Temperature (°C)", -15.0, 45.0, 25.0, 0.5, key=f"{key_prefix}_temp")
    hum = st.slider("Relative Humidity (%)", 10, 100, 50, key=f"{key_prefix}_hum")
    wind = st.slider("Wind Speed (km/h)", 0.0, 80.0, 15.0, 1.0, key=f"{key_prefix}_wind")

    st.write("")
    if st.button("🚀 Initialize Load Prediction", type="primary", use_container_width=True, key=f"{key_prefix}_btn"):
        st.session_state.user_inputs = {
            'target_dt': target_dt,
            'temp': temp, 'hum': hum, 'wind': wind
        }
        st.session_state.prediction_done = True
        st.rerun()


# ==========================================
# APP ROUTING: INITIAL SCREEN VS. DASHBOARD
# ==========================================

# Injecting CSS for clean text wrapping and styling
st.markdown("""
    <style>
    [data-testid="stMetricLabel"] { white-space: normal !important; word-wrap: break-word !important; line-height: 1.3; margin-bottom: 5px; }
    [data-testid="stMetricValue"] { white-space: normal !important; word-wrap: break-word !important; font-size: 1.8rem !important; line-height: 1.2; }
    [data-testid="stMetricValue"] > div { white-space: normal !important; word-wrap: break-word !important; }
    .project-info-box { background-color: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 20px; border-left: 4px solid #00cc96; }
    .stats-box { background-color: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 15px; border: 1px solid #334155; height: 100%;}
    </style>
""", unsafe_allow_html=True)

if not st.session_state.prediction_done:
    st.write("")
    st.write("")
    st.markdown("<h1 style='text-align: center;'>⚡ AI Grid Load Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray;'>Advanced forecasting powered by historical telemetry, climate clustering, and dimensionality reduction.</p>",
        unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            render_input_form("center")

else:
    inputs = st.session_state.user_inputs
    target_dt = inputs['target_dt']
    temp = inputs['temp']
    hum = inputs['hum']
    wind = inputs['wind']

    ref_dt = get_reference_datetime(target_dt)
    is_proxy_date = target_dt.year != ref_dt.year

    with st.sidebar:
        # Pushed directly to the top of the sidebar
        render_input_form("sidebar")

        st.divider()
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.prediction_done = False
            st.rerun()

    # ==========================================
    # SECTION 1: LIVE DIAGNOSTICS & PREDICTION
    # ==========================================
    st.title("📊 Real-Time Grid Diagnostics")

    if is_proxy_date:
        st.warning(
            f"Note: Current date ({target_dt.year}) exceeds the database limits. Anchoring prediction to the closest historical calendar match: **{ref_dt.strftime('%B %d, %Y')}**.")
    else:
        st.divider()

    with st.spinner('Aggregating data and executing machine learning pipeline...'):
        hour, day_of_week, month, day_of_year = target_dt.hour, target_dt.weekday(), target_dt.month, target_dt.timetuple().tm_yday
        is_weekend = 1 if day_of_week in [5, 6] else 0
        if month in [12, 1, 2]:
            season = 1
        elif month in [3, 4, 5]:
            season = 2
        elif month in [6, 7, 8]:
            season = 3
        else:
            season = 4

        lag_24h = get_historical_lag(ref_dt, 24)
        lag_168h = get_historical_lag(ref_dt, 168)
        weather_cluster = get_closest_weather_cluster(temp, hum, wind)
        cluster_name = WEATHER_PROFILES.get(weather_cluster, f"Profile #{weather_cluster}")

        features = np.array([[temp, hum, wind, hour, day_of_week, month, day_of_year, is_weekend, season, lag_24h,
                              lag_168h, weather_cluster]])
        features_scaled = scaler_model.transform(features)
        features_pca = pca_model.transform(features_scaled)

        distances, indices = knn_model.kneighbors(features_pca)
        neighbor_values = knn_model._y[indices][0]
        prediction = np.mean(neighbor_values)
        confidence_std = np.std(neighbor_values)
        lower_bound = int(prediction - confidence_std)
        upper_bound = int(prediction + confidence_std)

        col1, col2, col3 = st.columns([1, 1.15, 1.25])
        with col1:
            if is_proxy_date:
                st.metric(label=f"Historical Proxy Anchor ({ref_dt.year})", value=f"{int(lag_24h):,} MW")
                st.caption(
                    "🔍 **What this means:** Since live data isn't available for future dates, the AI anchors to the exact equivalent calendar day in our historical dataset.")
            else:
                st.metric(label="Baseline Anchor (T-24H)", value=f"{int(lag_24h):,} MW")
                st.caption(
                    "🔍 **What this means:** The actual grid load from exactly 24 hours prior before adjusting for today's specific weather.")

        with col2:
            st.metric(label="Climate Profile (K-Means)", value=cluster_name)
            st.caption(
                "🔍 **What this means:** The algorithm categorized the input weather into this specific climate cluster.")
        with col3:
            st.metric(label="AI Load Forecast Range", value=f"{lower_bound:,} - {upper_bound:,} MW")
            st.caption(
                f"🔍 **What this means:** The AI projects the demand to fall within this range. Exact mathematical average: **{int(prediction):,} MW**.")

        st.write("")
        col_gauge, col_chart = st.columns([1, 2])

        with col_gauge:
            st.subheader("🎯 Mean Demand Severity")
            if prediction < 30000:
                bar_color = "#00cc96"
            elif prediction < 40000:
                bar_color = "#ffa15a"
            else:
                bar_color = "#ef553b"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': lag_24h, 'increasing': {'color': "#ef553b"}, 'decreasing': {'color': "#00cc96"}},
                gauge={
                    'axis': {'range': [20000, 60000], 'tickwidth': 1},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [20000, 35000], 'color': 'rgba(0, 204, 150, 0.1)'},
                        {'range': [35000, 45000], 'color': 'rgba(255, 161, 90, 0.1)'},
                        {'range': [45000, 60000], 'color': 'rgba(239, 85, 59, 0.2)'}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50000}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(
                f"<p style='text-align: center; color: #94A3B8; margin-top: -20px;'>Estimated Range: <b>{lower_bound:,} MW</b> - <b>{upper_bound:,} MW</b></p>",
                unsafe_allow_html=True)

        with col_chart:
            st.subheader(f"📈 7-Day Historical Trajectory (Ref: {ref_dt.year})")
            history_df = fetch_historical_chart_data(ref_dt, days=7)
            fig_line = go.Figure()

            if not history_df.empty:
                fig_line.add_trace(go.Scatter(x=history_df['Datetime'], y=history_df['consumption'], mode='lines',
                                              name='Actual History', line=dict(color='gray', width=2)))

            fig_line.add_trace(go.Scatter(x=[ref_dt], y=[prediction], mode='markers', name='Mean Forecast Node',
                                          marker=dict(color='#00cc96', size=16, symbol='diamond',
                                                      line=dict(color='white', width=2))))

            fig_line.update_layout(
                xaxis_title="True Historical Timeline", yaxis_title="Power Consumption (MW)", hovermode="x unified",
                height=350, margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()
        st.subheader("🧠 Under the Hood: Engine Analytics & Project Overview")
        col_pca, col_about = st.columns([1, 1.5])

        with col_pca:
            st.metric(label="Dimensional Compression (PCA)", value=f"12 ➔ {pca_model.n_components_} Dims")
            st.caption(
                "🔍 **What this means:** Principal Component Analysis (PCA) mathematically compressed 12 overlapping variables into pure, independent data patterns to optimize the final KNN prediction speed.")

        with col_about:
            st.markdown("""
            <div class="project-info-box">
                <h4 style='margin-top: 0; color: #E2E8F0;'>🚀 About This Project</h4>
                <p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 0;'>
                This AI-powered grid forecasting tool was built to predict energy demand with high precision by analyzing complex patterns.
                <ul style='color: #94A3B8; font-size: 0.9rem; margin-top: 5px;'>
                    <li><b>Data Pipeline:</b> Aggregated over 80,000 hours of historical energy consumption and meteorological telemetry.</li>
                    <li><b>Feature Engineering:</b> Extracted dynamic time features and anchoring historical lags (T-24H & T-168H).</li>
                    <li><b>Machine Learning:</b> Utilized <b>K-Means</b> for unsupervised climate profiling, <b>PCA</b> for dimensionality reduction, and an optimized <b>KNN Regressor</b> for the final load estimation.</li>
                </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # SECTION 2: DATA ANALYTICS & STATISTICS
    # ==========================================
    st.write("")
    st.write("")
    st.write("")
    st.markdown(
        "<h1 style='text-align: center; border-top: 2px solid #334155; padding-top: 40px;'>📚 Statistical Inference & Anomalies</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray; margin-bottom: 40px;'>Mathematical verification of data distributions and operational extremes based on the training data.</p>",
        unsafe_allow_html=True)

    with st.spinner("Executing statistical tests and loading comprehensive data..."):
        df_stats = get_full_stats_data()

        # --- Quick Stats Row ---
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Total Data Points", f"{len(df_stats):,}")
        col_stat2.metric("Max Demand Recorded", f"{int(df_stats['consumption'].max()):,} MW")
        col_stat3.metric("Min Demand Recorded", f"{int(df_stats['consumption'].min()):,} MW")
        col_stat4.metric("Average Temperature", f"{round(df_stats['temperature_c'].mean(), 1)} °C")

        st.write("")

        col_inf1, col_inf2 = st.columns(2)

        # 1. T-Test (Weekend vs Weekday)
        with col_inf1:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.markdown("#### 1. Hypothesis Testing (T-Test)")
            st.markdown(
                "Testing if there's a statistically significant difference in power consumption between Weekdays and Weekends.")

            weekend_data = df_stats[df_stats['is_weekend'] == 1]['consumption']
            weekday_data = df_stats[df_stats['is_weekend'] == 0]['consumption']
            t_stat, p_val = stats.ttest_ind(weekday_data, weekend_data, equal_var=False)

            st.write(f"• **Avg Weekday Load:** {weekday_data.mean():,.0f} MW")
            st.write(f"• **Avg Weekend Load:** {weekend_data.mean():,.0f} MW")
            st.write(f"• **P-Value:** `{p_val:.4e}`")

            if p_val < 0.05:
                st.success(
                    "✅ **Conclusion:** The P-Value is < 0.05. There is a statistically significant drop in power consumption during weekends.")
            else:
                st.info("ℹ️ **Conclusion:** No significant statistical difference found.")
            st.markdown('</div>', unsafe_allow_html=True)

        # 2. Confidence Interval
        with col_inf2:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.markdown("#### 2. Population Mean (95% CI)")
            st.markdown(
                "Calculating the 95% Confidence Interval for the true historical mean of the grid's power consumption.")

            data_clean = df_stats['consumption'].dropna()
            mean_val = np.mean(data_clean)
            std_error = stats.sem(data_clean)
            ci = stats.t.interval(0.95, df=len(data_clean) - 1, loc=mean_val, scale=std_error)

            st.write(f"• **Sample Mean:** {mean_val:,.0f} MW")
            st.write(f"• **Lower Bound:** {ci[0]:,.0f} MW")
            st.write(f"• **Upper Bound:** {ci[1]:,.0f} MW")
            st.info(
                f"✅ **Conclusion:** We are 95% confident that the true population mean of power consumption lies strictly between {ci[0]:,.0f} MW and {ci[1]:,.0f} MW.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")

        # 3. Anomaly Detection (Z-Score)
        st.markdown("#### 3. Anomaly Detection (Z-Score > 3.0)")
        df_stats['z_score'] = np.abs(stats.zscore(df_stats['consumption']))
        outliers = df_stats[df_stats['z_score'] > 3.0]

        st.markdown(
            f"Detected **{len(outliers):,} statistical anomalies** (|Z| > 3) out of {len(df_stats):,} historical hours.")

        if len(outliers) > 0:
            with st.expander("🚨 View Top 5 Most Extreme Historical Grid Events"):
                top_outliers = outliers.sort_values(by='z_score', ascending=False).head(5)
                display_outliers = top_outliers[
                    ['Datetime', 'consumption', 'temperature_c', 'season_name', 'z_score']].copy()
                display_outliers['Datetime'] = display_outliers['Datetime'].dt.strftime('%Y-%m-%d %H:%00')
                display_outliers['consumption'] = display_outliers['consumption'].apply(lambda x: f"{x:,.0f} MW")
                display_outliers['temperature_c'] = display_outliers['temperature_c'].apply(lambda x: f"{x:.1f} °C")
                display_outliers['z_score'] = display_outliers['z_score'].apply(lambda x: f"{x:.2f} Z")
                display_outliers.columns = ['Date & Time', 'Extreme Load', 'Temperature', 'Season', 'Z-Score']
                st.dataframe(display_outliers, use_container_width=True, hide_index=True)