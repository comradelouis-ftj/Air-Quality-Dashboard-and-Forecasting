# Air Quality Forecasting and Dashboard Project
---
A web application for dashboarding and forecasting PM 2.5 readings in London, on five different locations in London. This is achieved by extracting data from open source APIs for both the PM 2.5 readings and weather readings over the past three years (2023 - 2026), whose data is then processed and stored inside a local postgresql database for further processing inside the dashboard or for machine learning purposes. The output of the project is, as such, a web application containing both dashboard and forecasting features for air quality in London over the past three years.

---

## 📌 Local Set-Up Requirements

1. Create a virtual environment by running: 'python -m venv venv'
2. Install all proper dependencies (use 'pip install -r requirements.txt').
3. Make sure that postgresql is installed in order to store the datasets.
4. Create a .env file which should store your OpenAQ API Key ([login & check key here](https://explore.openaq.org/login?redirect=/login)), postgresql password, and postgresql username
4. Run these .ipynb files (in this order): data_extraction.ipynb, data_cleaning.ipynb, and data_storage_db.ipynb

**Note:** 
- It should be known that the .ipynb files can be editted, its date intervals and other parameters utilized can be editted so the extracted data is more recent. Make sure to check out the functions utilized in these notebooks on the functions folder.
- Make sure that you have an OpenAQ API account to get its free API key. Note that the free API keys does have limitations, so do check its documentation before using it

---

## 📖 Project Background

The idea behind this project is to practice and apply different stages of data engineering to feed a reasonably complete data for primarily machine learning purposes. These stages include ingestion, transformation, and serving. Doing this would allow the machine learning process to simply extract the specific portions of weather and air quality readings for modelling, as such simplifying the data extraction process for the machine learning process (of course, the functions onlyhandle basic data preprocessing, i.e. handling missing records for specific categorical features, and as such, more specific preprocesisng steps has to be done inside the modelling notebook). Based on this background information, the key components of this project includes:
1. Data engineering functions: applies ingestion and transformation of weather and air quality readings using python functions, then serves the cleaned dataset for machine learning purposes.
2. Machine learning: applies sn LSTM time-series model for forecasting air quality, specifically PM 2.5 levels, for the next 3 hours.
3. Web application: utilizing streamlit for building and deploying a web application for showing weather and air quality dashboard as well as air wuality forecasting.

---

## 📱 Key Web App Features and Functionalities

1. **Dashboard Page**: A dashboard showing details on weather condition and a summary of PM 2.5 levels across an interval of time, aggregated by hour. The specific region and time interval may be chosen by the user via a dropdown selection bar in the dshboard page.
2. **Prediction Page**: A streamlit form which allows users to select a specific region in London using a dropdown selection bar. When selected, the app would forecasts for PM 2.5 readings for the next three hours.

---

## 🧠 Model Layer Details
1. **Input Layer:** Composed of 2 different layers, numerical input (past PM 2.5 and weather readings) and categorical input (forecast horizon and location id).
2. **Numerical Model Layer**: Passes numerical values into a Conv1d layer to extract local patterns, which is then passed into an LSTM & attention layer (to learn historical patterns), and then normalized then applied to a global max pooling layer to only highlight certain units/neurons with relatively high spikes.
3. **Categorical Embeddings Layer**: Processes static features (location id and forecast horizon) using two embedding layers (one for each features), which is then combined for later processing.
4. **Output Layer**: Consists of Dense layers, which first concatenates (combines) outputs from the numerical model and categorical embedding layers. There are two Dense layers, and in between there is a dropout layers to stabilize the outputs, which is fed to the latter dense layer. This last dense layer is the one that outputs the forecast.

Note: further details could be viewed in the modelling.ipynb notebook, on the modelling section

---
## 📊 Data Engineering Functions

Functions for the data engineering process can be viewed in the functions folder. The python files within said folder each has different purposes:
1. **database_actions_functions.py**: used to handle any actions that query directly to the postgresql database, including inserting new values, initializing the database, and extracting a set of values from the database.
2. **ingestion_functions.py**: used to extract weather and air quality readings from the api, storing the results in .csv files located in the datasets_raw folder.
3. **transformatuon_functions.py**: cleans, combines, and aggregate the weather and air quality readings into a format legible for the machine learning process, the outputs of the functions within this file is used as a basis for insertion/serving, the cleaned version of the dataset that is generated by this file is also used as the dataset stored inside the postgresql database.

Note: the function/usage of the functions within these files can be viewed in the **data_extraction.ipynb, data_cleaning.ipynb, and the data_storage_db.ipynb** notebooks.

---
## Directory Description

```
.
├── .gitignore
├── app_streamlit/
│   ├── pages/
│   │   ├── dashboard.py
│   │   └── predict.py
│   ├── utils/
│   │   └── functions.py
│   └── main.py
├── automate_update.bat
├── data_cleaning.ipynb
├── data_extraction.ipynb
├── data_storage_db.ipynb
├── datasets_aggr/
│   └── weather_readings_mlready.csv
├── datasets_clean/
│   ├── sensor_stationid_pair.csv
│   ├── station_details.csv
│   └── weather_pm25_readings.csv
├── datasets_raw/
│   ├── sensor_readings.csv
│   ├── station_sensors.csv
│   └── weathers_dataset.csv
├── docs/ ## project descriptions
│   ├── Data Model.png
│   ├── directory_tree.txt
│   ├── README.md
│   └── Transformation Schema.png
├── extraction_uptodate_dataset.py
├── functions/ ## stores data pipeline functions + scalers
│   ├── scalers/ ## stores scalers used for modelling
│   │   ├── 1/
│   │   │   ├── actual_pm25_scaler.joblib
│   │   │   ├── hour_scaler.joblib
│   │   │   ├── pm25_rolling_mean_6h_scaler.joblib
│   │   │   ├── pm25_scaler.joblib
│   │   │   ├── rain_scaler.joblib
│   │   │   ├── relative_humidity_2m_scaler.joblib
│   │   │   ├── surface_pressure_scaler.joblib
│   │   │   ├── temperature_2m_scaler.joblib
│   │   │   └── wind_speed_10m_scaler.joblib
│   │   ├── 2/
│   │   ├── 3/
│   │   ├── 4/
│   │   └── 5/
│   ├── database_actions_functions.py
│   ├── ingestion_functions.py
│   └── transformation_functions.py
├── modelling.ipynb
└── models/ ## stores models
│   ├── lstm_best.keras
│   └── training_logs_lstm.csv
└── requirements.txt
```