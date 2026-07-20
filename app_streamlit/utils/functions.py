import tensorflow as tf
import joblib

import pandas as pd

import altair as alt

import os

# Function for feature Scaling
def scale_features(df_input):
    code_station = df_input['station_id'].iloc[0] # extracting the dataframe's station
    path_station_scaler = os.path.join(os.getcwd(), os.path.abspath(f'functions/scalers/{int(code_station)}')) # path for a statuib's scaler
    scaler_target = None # stores the target feature's scaler (actual_pm25)

    # Loops through all files in the scaler folder
    for file_name in os.listdir(path_station_scaler):
        col = file_name.split('_scaler')[0] # sxtracts the scaler's intended feature
        scaler = joblib.load(os.path.join(path_station_scaler, file_name)) # loading the scaler
        if 'actual' in file_name:
            # If the scaler is the target feature's scaler, it is stored in the scaler_target variable
            scaler_target = scaler
            continue

        # transforming the numerical features (aside from target feature)
        df_input[col] = scaler.transform(df_input[col].values.reshape(-1, 1))
        
    return df_input, scaler_target # returns the scaled dataframe and the target scaler

# Function for Encoding Stations
def apply_encoding(x):
    if x=='Camden Kerbside':
        return 1
    elif x=='Greenwich':
        return 2
    elif x=='London Honor Oak Park':
        return 3
    elif x=='Waterloo Place':
        return 4
    elif x=='Brent':
        return 5

# Function for Loading Model & Model Inference
def predict_pm25(df_input, horizon, cols_num, cols_cat):
    model = tf.keras.models.load_model(os.path.abspath('./models/lstm_best.keras'), safe_mode=False) # extracts model
    
    df_try = df_input.copy() # creates copy of input data
    df_try = df_try[cols_num+cols_cat] # extracts neccessary columns
    df_try['horizon'] = horizon # extracts the horizon
    df_try, scaler_target = scale_features(df_try) # scaling the numerical features and extracting target scaler

    # Creating the proper input for model inference 
    # The shape of numerical input: (None, 20, 6) 
    # the shape of categorical input: (None, 20, 2)
    input_num = df_try[cols_num].values.reshape(len(df_try), len(cols_num))[None, :, :]
    input_cat = df_try[cols_cat+['horizon']].values.reshape(len(df_try), 2)[None, :, :]

    pred = model.predict([input_num, input_cat]) # model inference
    return scaler_target.inverse_transform(pred)[0][0] # returning the actual, non-scaled PM 2.5 level

# Function for Assembling 3-Hour Forecasts
def get_forecast(df_input, station_name):
    df_cp = df_input.copy() # creates a copy so the original dataframe does not get transformed
    df_cp['station_id'] = df_cp['station_name'].apply(apply_encoding) # encodes station id to singular digits
    cols_num = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure', 'rain', 'pm25'] # numerical columns for model inference
    cols_cat = ['station_id'] # categorical column for model inference

    df_station_all = df_cp[df_cp['station_name']==station_name].copy() 
    df_station_all.reset_index(drop=True, inplace=True)
    df_station = df_station_all.iloc[-20:].copy() # takes the last 20 weather records
    df_station_n = df_station[['time_reading']+cols_num+cols_cat] # extracts the necessary columns + the timestamp
    df_station_n['label'] = 'original' # creates a label to denote that the current PM2.5 values are the original values from the database

    if df_station_n['pm25'].isna().mean()>=0.70:
        # If most of the last 20 entries are missing, the model will not make an inference
        return None, 'Too much of the data is missing, try again with different region or try again later'
    elif df_station_n['pm25'].isna().mean()>0:
        # if the number of missing values are still somewhat acceptable, the model will fill in the missing values with its inference, and use a 
        # combination of actual, pre-existing values and the model's inference to make the forecast
        for v in range(len(df_station_n)):
            curr_pm25 = df_station_n.iloc[v] # extracts the record
            idx = curr_pm25.name # extracts the index
            
            if pd.isna(curr_pm25.pm25):
                # Case when the PM 2.5 value is missing
                start = (idx)-20
                input_vals = df_station_all.iloc[start-1:idx-1] # takes the previous 20 records for inference
                input_vals['pm25'] = input_vals['pm25'].interpolate(method='polynomial', order=2).ffill().bfill() # if some of the previous 20 values are missing, it is interpolated
                pred = predict_pm25(input_vals, cols_num=cols_num, cols_cat=cols_cat, horizon=1) # inference
                
                # Filling in the new values
                df_station_n.loc[idx, 'pm25'] = pred
                df_station_n.loc[idx, 'label'] = 'model-augmented (due to missing value)' # giving the augmented values a different label
                df_cp.loc[idx, 'pm25'] = pred

    # Model 3-hour forecast for PM 2.5 readings
    forecasts = {
        'pm25': [],
        'time': [],
        'label': []
    }
    for i in range(3):
        # Horizon is 3 hours, so each loop forecasts for a different horizon
        pred_3h = predict_pm25(df_station_n, cols_num=cols_num, cols_cat=cols_cat, horizon=i+1)
        time_curr = df_station['time_reading'].max() + pd.Timedelta(hours=i+1)
        forecasts['pm25'].append(pred_3h)
        forecasts['time'].append(time_curr)
        forecasts['label'].append('Forecast')

    df_station_n = df_station_n.rename(columns={
        'time_reading': 'time'
    })

    return pd.DataFrame(forecasts), df_station_n[['time', 'label']+cols_num+cols_cat] # returns forecasts and the last 20 records of the dataframe

# Function for Visualizing Forecast result
def create_chart(raw_data, forecasts):
    result = pd.concat([raw_data, forecasts])
    result_n = result.sort_values(by='time').reset_index(drop=True)
    result_n['block_id'] = (result_n['label'] != result_n['label'].shift()).cumsum()

    extended_rows = []
    for i in range(1, len(result_n)):
        if result_n.loc[i, 'label'] != result_n.loc[i-1, 'label']: # checks if the current label is different from the previous
            boundary_row = result_n.loc[i-1].copy() # takes the previous value
            boundary_row['label'] = result_n.loc[i, 'label'] # replaces the the previous row's copy's label with the current label
            boundary_row['block_id'] = result_n.loc[i, 'block_id'] # replaces the previous row's copy's block id with the current one
            extended_rows.append(boundary_row) # inserts the boundary row into the list

    result_connected = pd.concat([result_n, pd.DataFrame(extended_rows)], ignore_index=True).sort_values('time') # combines the extended rows and the result dataframe, sorting it by time

    # Building streamlit-compatible altair chart
    chart = alt.Chart(result_connected).mark_line(
        point={'size': 150, 'filled': True}
    ).encode(
        x=alt.X('time:T', title='Time'),
        y=alt.Y('pm25:Q', title='PM2.5'),
        color=alt.Color(
            'label:N', 
            scale=alt.Scale(
                domain=['original', 'model-augmented (due to missing value)', 'Forecast'],
                range=['blue', 'purple', 'green']
            ),
            legend=alt.Legend(title="Category Layers")
        ),
        detail='block_id:N', # prevents the lines from overlapping with each other
        tooltip=[
            alt.Tooltip('time:T', title='Timestamp', format='%Y-%m-%d %H:%M:%S'),
            alt.Tooltip('pm25:Q', title='PM2.5', format='.2f'),
            alt.Tooltip('label:N')
        ]
    ).properties(
        height=450, width=800
    ).interactive() # allow interactiveness

    return chart