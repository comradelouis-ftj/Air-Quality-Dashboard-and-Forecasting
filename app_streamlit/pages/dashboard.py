import pandas as pd
import os

import streamlit as st

st.set_page_config(
    page_title='London Air Quality & Weather',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# -----------------------------------------------------------------------------
# 1. Data Caching & Extraction
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['time_reading'] = pd.to_datetime(df['time_reading'])
    return df

try:
    df = load_data('https://raw.githubusercontent.com/comradelouis-ftj/Air-Quality-Dashboard-and-Forecasting/refs/heads/master/datasets_aggr/weather_readings_mlready.csv')
except:
    st.error("Error: 'weather_readings_mlready.csv' not found")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Setting Title
# -----------------------------------------------------------------------------
st.title('London Air Quality & Weather Dynamics Dashboard')
st.divider()

# -----------------------------------------------------------------------------
# 3. Station and Datetime Selection
# -----------------------------------------------------------------------------
max_time = df['time_reading'].max()
min_time = df['time_reading'].min()

with st.form('Filter Form'):
    default_start = (max_time - pd.Timedelta(hours=24)).date()
    default_end = max_time.date()

    col1, col2 = st.columns([1, 1])
    with col1: # Station selection
        selected_station = st.selectbox(
            "Select Region / Station Name:",
            options=sorted(['Waterloo Place', 'Camden Kerbside', 'Greenwich', 'London Honor Oak Park', 'Brent']), #df['station_name'].unique()
            index=0
        )

    with col2: # Date interval selection
        selected_date_range = st.date_input(
            "Select Date Range:",
            value=(default_start, default_end),
            min_value=min_time.date(),
            max_value=max_time.date()
        )
    
    click_show = st.form_submit_button("Show", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Filtering Data
# -----------------------------------------------------------------------------
if not click_show:
    st.info("💡 Select region and date, then click **Show** to render the weather and PM 2.5 readings.")    
else:
    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
    else:
        start_date = end_date = selected_date_range

    # Filtering dataframe
    filtered_df = df[
        (df['station_name'] == selected_station) & (df['time_reading'].dt.date >= start_date) & (df['time_reading'].dt.date <= end_date)
    ].sort_values(by='time_reading')

    # Prepare chart-friendly indexed dataframes
    chart_df = filtered_df.set_index('time_reading')

    st.divider()

    # -----------------------------------------------------------------------------
    # 5. Displaying Charts
    # -----------------------------------------------------------------------------
    if filtered_df.empty:
        st.warning("No data found for the selected station and timestamp, please try again")
    else:
        # PM 2.5 Air Quality Distribution
        with st.container(border=1):
            st.subheader("📊 PM 2.5 Air Quality Distribution")
            
            m1, m2, m3 = st.columns(3, border=1)
            avg_pm = filtered_df['pm25'].mean()
            max_pm = filtered_df['pm25'].max()
            
            m1.metric("Average PM 2.5 Concentration", f"{avg_pm:.2f} µg/m³" if not pd.isna(avg_pm) else "N/A")
            m2.metric("Peak PM 2.5 Level", f"{max_pm:.2f} µg/m³" if not pd.isna(max_pm) else "N/A")
            m3.metric("Total Recorded (Hours)", f"{len(filtered_df)} hrs")
            
            # PM 2.5 Trend Line Chart
            st.markdown(f"**PM 2.5 Distribution - ({start_date} to {end_date})**", text_alignment='center')
            st.line_chart(chart_df['pm25'], y_label="PM 2.5 (µg/m³)")

        st.markdown("---")

        # Temperature & Relative Humidity Plots
        with st.container(border=1):
            st.subheader("🌡️ Ambient Temperature & Relative Humidity Profiles")
            
            t_col, h_col = st.columns(2)
            with t_col:
                st.markdown("**Temperature Distribution (°C)**", text_alignment='center')
                st.line_chart(chart_df['temperature_2m'], color="#FF4B4B", y_label="Temperature (°C)")
            with h_col:
                st.markdown("**Relative Humidity Levels (%)**", text_alignment='center')
                st.line_chart(chart_df['relative_humidity_2m'], color="#0068C9", y_label="Humidity (%)")
            st.markdown(f"**({start_date} to {end_date})**", text_alignment='center')

        st.markdown("---")

        # Wind Speed Plots
        with st.container(border=1):
            st.subheader("💨 Wind Speed Distribution")
            st.line_chart(chart_df['wind_speed_10m'], color="#29B6F6", y_label="Wind Speed (km/h)")
            st.markdown(f"**({start_date} to {end_date})**", text_alignment='center')

        st.markdown("---")

        # Atmospheric Pressure and Rainfall
        with st.container(border=1):
            st.subheader(f"🌦️ Atmospheric Pressure & Rainfall Influx")
            st.markdown(f"**({start_date} to {end_date})**", text_alignment='center')
            
            p_col, r_col = st.columns(2)
            with p_col:
                st.markdown("**Surface Pressure Trends (hPa)**", text_alignment='center')
                st.line_chart(chart_df['surface_pressure'], color="#7D3C98", y_label="Pressure (hPa)")
            with r_col:
                st.markdown("**Precipitation / Rain Accumulation (mm)**", text_alignment='center')
                st.area_chart(chart_df['rain'], color="#1ABC9C", y_label="Rainfall (mm)")