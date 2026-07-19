import pandas as pd
import math

import os

def merge_weather_pm25_reads(pm25_readings=None, weather_readings=None, folder_name='datasets_raw'):
    if pm25_readings is None and weather_readings is None: # condition if no actual dataframe
        # Getting weather & PM 2.5 readings directories
        path_datasets_raw = os.path.join(os.getcwd(), folder_name)
        pm25_readings = pd.read_csv(os.path.join(path_datasets_raw, 'sensor_readings.csv'))
        weather_readings = pd.read_csv(os.path.join(path_datasets_raw, 'weathers_dataset.csv'))

    # Conversion of date features to proper datetime format for merging
    weather_readings['time'] = pd.to_datetime(weather_readings['time'].apply(lambda x: x.replace('T', ' ')))
    pm25_readings['timestamp_from'] = pd.to_datetime(pm25_readings['timestamp_from'].apply(lambda x: x.replace('T', ' ').replace('Z', '')))
    pm25_readings['timestamp_to'] = pd.to_datetime(pm25_readings['timestamp_to'].apply(lambda x: x.replace('T', ' ').replace('Z', '')))

    # Merging by stations & timestamp, note that merge is done on timestamp_from, as it is assumed on these features, both APIs denote the
    # starting time when a reading is taken
    merged_weather_pm25 = weather_readings.merge(
        pm25_readings,
        how='left',
        left_on=['time', 'station_id_london'],
        right_on=['timestamp_from', 'location_id']
    )
    
    return merged_weather_pm25 # returns a merged dataframe

# Function for Extracting Information on Data Inconsistencies
def check_inconsistencies(df_merged):
    # checks the number of unique sensor Ids and station Ids per station name, if the number is not one, then further processing is needed
    check_inconsitency = df_merged.groupby(by='station_name').agg({
        'sensor_id': 'nunique',
        'station_id_london': 'nunique',
        'location_id': 'nunique'
    })
    inconsitencies_more_than_1 = (check_inconsitency>1).any(axis=1) # stations with >1 station id, sensor id, or location id
    inconsitencies_less_than_1 = (check_inconsitency<1).any(axis=1) # stations with >1 station id, sensor id, or location id

    # Checking inconsistent station names (does not include cases when the PM 2.5 readings' station names are missing, as this is caused by sensor errors/delays)
    num_inconsistent_station_name = len(df_merged[(df_merged['location_name'] != df_merged['station_name']) & (df_merged['location_name'].isnull().any()==False)])

    # Checking duplicates & missing values:
    duplicates = df_merged.duplicated().sum()
    missing = df_merged.isnull().sum()

    # Prints out the details
    print(f'On merged dataset found {duplicates} duplicated records, missing value details: \n{missing}')
    print(f'Stations with > 1 sensors sending values? {inconsitencies_more_than_1.any()}')
    print(f'Stations with 0 sensors sending values? {inconsitencies_less_than_1.any()}')
    print(f'Stations with inconsistent names? {num_inconsistent_station_name}')
    return inconsitencies_less_than_1, inconsitencies_more_than_1, num_inconsistent_station_name, duplicates, missing # returns all the details

# Function for Filling in Non-Numerical Data Points
def fill_nan_values(station_name: str, df_merged):
    df_current = df_merged[df_merged['station_name']==station_name].copy() # takes readings on a specific station

    # Since the weather reading's data is complete (it is used as the SSL) and due to the fact that the missing values all come from the PM2.5 
    # readings,any missing station id, location name, and timestamp is replaced by the weather reading counterpart
    df_current['location_id'] = df_current['station_id_london']
    df_current['location_name'] = df_current['station_name']
    df_current['timestamp_from'] = df_current['time']

    # For the sensor ID, if the station's readings utilize only 1 sensor, then the sensor id is simply made uniform for the whole dataset
    list_unique_sensor_id = [i for i in list(df_current['sensor_id'].unique()) if not math.isnan(i)]
    if len(list_unique_sensor_id)==1:
        df_current['sensor_id'] = int(list_unique_sensor_id[0])

    df_current['weather_id'] = 'W' + df_current['sensor_id'].astype(int).astype(str) + '_' + df_current['time'].dt.strftime('%Y%m%d%H') # creating unique key for each weather reading

    # The timestamp_to feature is replaced, as it is redundant and is already assumed that each reading encompasses 1 hour of data
    df_current.drop(columns=['timestamp_to'], inplace=True) 
    return df_current

# Function for Extracting Cleaned Dataset
def clean_extracted_dataset(pm25_readings=None, weather_readings=None, folder_name='datasets_raw'):
    merged_weather_pm25 = merge_weather_pm25_reads(pm25_readings, weather_readings, folder_name) # merging PM2.5 & weather readings
    print(f'Condition before cleaning:')
    _ = check_inconsistencies(merged_weather_pm25) # showing characteristics on merged data - pre cleaning

    # Looping through all unique stations and filling in its missing non-numerical data
    unique_stations = list(merged_weather_pm25['station_name'].unique())
    dict_df = {} # stores each station's clean dataset
    for station in unique_stations:
        dict_df[station] = fill_nan_values(station, merged_weather_pm25)

    # Merge all station's datasets, whose missing non-numeric values have been replaced
    merged_weather_pm25_new = pd.concat([dict_df[station] for station in dict_df], axis=0, ignore_index=True)

    print(f'\nCondition after cleaning:')
    _ = check_inconsistencies(merged_weather_pm25_new) # showing characteristics on merged data - post cleaning

    return merged_weather_pm25_new # returns cleaned dataset

# Function for Saving & Normalizing PM2.5 and Weather Readings
def normalize_save_pm25_weather_readings(folder_name_raw='datasets_raw', folder_name_clean = 'datasets_clean'):
    # Extracting clean dataset directory
    path_clean_dataset = os.path.join(os.getcwd(), folder_name_clean)
    if not os.path.exists(path_clean_dataset):
        os.makedirs(path_clean_dataset)

    merged_weather_pm25 = clean_extracted_dataset(folder_name=folder_name_raw) # extracting merged and cleaned weather & PM2.5 readings

    # Getting station details information
    path_datasets_raw = os.path.join(os.getcwd(), folder_name_raw)
    df_station_sensors = pd.read_csv(os.path.join(path_datasets_raw, 'station_sensors.csv'))

    # Creating a new dataset only containing the sensor id - station id pair
    id_station_detail = df_station_sensors[['sensor_id', 'station_id']]
    id_station_detail.drop_duplicates(inplace=True, ignore_index=True)

    # Creating a new dataset with only details for each station (name & coordinates)
    station_details = df_station_sensors[['station_id', 'station', 'long', 'lat']].rename(columns={
        'station': 'station_name'
    })
    station_details.drop_duplicates(inplace=True, ignore_index=True)

    # Creating a new weather & PM2.5 reading dataset, with all redundant features or unneessary features (i.e. features already present or could be accessed
    # via connection to other datasets) removed, including the station details, and the duplicated timestamp feature
    # This is done to ensure no transitive/partial dependencies exist, thus enabling 3NF normalization
    merged_weather_pm25.drop(columns=['station_id_london', 'station_name', 'long', 'lat', 'location_id', 'location_name', 'timestamp_from'], inplace=True)
    merged_weather_pm25 = merged_weather_pm25[['weather_id', 'sensor_id', 'time', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure', 'rain', 'values_pm25']]

    # Saving normalized dataset to CSV format
    id_station_detail.to_csv(os.path.join(path_clean_dataset, 'sensor_stationid_pair.csv'), index=False)
    station_details.to_csv(os.path.join(path_clean_dataset, 'station_details.csv'), index=False)
    merged_weather_pm25.to_csv(os.path.join(path_clean_dataset, 'weather_pm25_readings.csv'), index=False)

    return id_station_detail, merged_weather_pm25, station_details # returns all the normalized dataframes

# Shortened Function for Saving & Normalizing Up-to-Date PM2.5 and Weather Readings
def normalize_save_pm25_weather_readings_short(pm25_readings, weather_readings):
    merged_weather_pm25 = clean_extracted_dataset(pm25_readings, weather_readings) # extracting merged and cleaned weather & PM2.5 readings

    # Creating a new weather & PM2.5 reading dataset, with all redundant features or unneessary features (i.e. features already present or could be accessed
    # via connection to other datasets) removed, including the station details, and the duplicated timestamp feature
    # This is done to ensure no transitive/partial dependencies exist, thus enabling 3NF normalization
    merged_weather_pm25.drop(columns=['station_id_london', 'station_name', 'long', 'lat', 'location_id', 'location_name', 'timestamp_from'], inplace=True)
    merged_weather_pm25 = merged_weather_pm25[['weather_id', 'sensor_id', 'time', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure', 'rain', 'values_pm25']]

    return merged_weather_pm25 # returns all the normalized dataframes