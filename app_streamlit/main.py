import streamlit as st

st.set_page_config(
    page_title='London Air Quality & Weather',
    layout='wide',
    initial_sidebar_state='collapsed'
)

dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True)
prediction = st.Page("pages/predict.py", title="Prediction", icon="🔮")

pg = st.navigation({
    "Dashboard": [dashboard],
    #"Machine Learning": [prediction]
})
pg.run()