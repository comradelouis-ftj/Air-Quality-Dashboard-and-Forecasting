import pandas as pd
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
    df['time_reading'] = pd.to_datetime(df['time_reading'])
    df['pm25_rolling_mean_6h'] = df['pm25'].rolling(window=6, min_periods=1).mean()
    df['hour'] = df['time_reading'].dt.hour
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
