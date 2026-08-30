import pandas as pd
import numpy as np
import os

from utils.functions import get_forecast, create_chart

import streamlit as st

st.set_page_config(
    page_title='London Air Quality & Weather',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# -----------------------------------------------------------------------------
# 1. DATA INGESTION & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    list_shortened_dfs = []
    moving_avg = {
        'temperature_2m': [11, 24],
        'relative_humidity_2m': [9, 24],
        'wind_speed_10m': [24],
        'surface_pressure': [24],
        'rain': [5],
        'pm25': [24]
    }

    for station in df['station_name'].unique():
        current = df[df['station_name']==station].tail(1000)
        current['time_reading'] = pd.to_datetime(df['time_reading'])
        current['hour_sin'] = np.sin((2*np.pi*current['time_reading'].dt.hour)/24)
        current['hour_cos'] = np.cos((2*np.pi*current['time_reading'].dt.hour)/24)

        for col in moving_avg:
            for val in moving_avg[col]:
                current[f'{col}_rolling{val}'] = current[col].rolling(val).mean()
        list_shortened_dfs.append(current)

    df = pd.concat(list_shortened_dfs, ignore_index=True)
    return df

try:
    df = load_data('https://raw.githubusercontent.com/comradelouis-ftj/Air-Quality-Dashboard-and-Forecasting/refs/heads/master/datasets_aggr/weather_readings_mlready.csv')
except:
    st.error("Error: 'weather_readings_mlready.csv' not found")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Setting Title
# -----------------------------------------------------------------------------

st.title('London PM 2.5 Rates Forecast')
st.divider()

# -----------------------------------------------------------------------------
# 3. Creating Forecasting Form
# -----------------------------------------------------------------------------

forecasts, raw_data = None, None

col1, col2 = st.columns([1, 1], border=1)
with col1:
    with st.form('Select Station'):
        selected_station = st.selectbox(
            "Select Region / Station Name:",
            options=sorted(['Waterloo Place', 'Camden Kerbside', 'Greenwich', 'London Honor Oak Park', 'Brent']),
            index=0
        )
        click_show = st.form_submit_button("Show", use_container_width=True)
with col2:
    if not click_show:
        st.info('💡 Select region, then click show to show 3-hour forecast from the most recent timestamp')
    else:
        forecasts, raw_data = get_forecast(df.copy(), selected_station)

        if forecasts is None:
            st.error(f'**⚠️ {raw_data}**')
        else:
            st.markdown(f'**6-Hour Forecast**', text_alignment='center')
            st.table(forecasts)

# -----------------------------------------------------------------------------
# 4. Plotting Forecast
# -----------------------------------------------------------------------------
if click_show:
    if forecasts is None:
        pass
    else:
        with st.container(border=1):
            st.markdown(f'**PM 2.5 Forecast - {selected_station}**', text_alignment='center')
            chart = create_chart(raw_data, forecasts)
            st.altair_chart(chart, width='stretch')
