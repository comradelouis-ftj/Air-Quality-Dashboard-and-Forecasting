# API extraction
import requests

# Concurrent processing & automation
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# Date manipulation
import dateutil
import datetime
import time

# Data processing - for csv conversion
import pandas as pd

# for file manipulation & environment variable
import os
from dotenv import load_dotenv

load_dotenv() # load environment variables
thread_local = threading.local() # Initializing thread local storage
headers = {
    'X-API-Key': os.getenv('API_OPENAQ') # initializing request header
}


# ################################################ STATION SENSORS EXTRACTOR FUNCTIONS ################################################

# Function for API call to get sensor ids per location
def extract_sensor_ids(station_id: int, station_name: str, headers=headers):
    sensors = requests.get(f'https://api.openaq.org/v3/locations/{station_id}', headers=headers) # sends in API call
    stations_details = {} # stores station's name, id, coordinates, and sensors
    sensors_list = [] # stores a station's list of sensors id
    # Each API call returns many attributes, of which only the station's name, id, coordinates, and a list of its sensors (which sends PM 2.5 readings) is taken
    # in order to store rudimentary station details as well as to allow later API calls using said coordinates to extract each location's weather from the 
    # earlier mentioned time period

    print(f'Getting sensors in {station_name}')
    result = sensors.json()['results'][0] # get API results

    # Extracts all the aforementioned attributes from API call result
    for v in result['sensors']:
        if v['name'].startswith('pm25'):
            sensors_list.append(v['id'])
            #print(v)
    stations_details['sensors']=sensors_list
    stations_details['long']=result['coordinates']['longitude']
    stations_details['lat']=result['coordinates']['latitude']
    stations_details['id']=station_id
    stations_details['name']=station_name
    #print(f'Sensors at {station_name} location: long {long}, lat {lat})\n')

    return stations_details # returns a dictionary of the aforementioned values

# Function to execute sensor ID extraction
def get_station_details(list_needed_stations: dict, worker=6):
    station_details_full = {} # stores a dictionary of locations and its name, id, coordinates, and sensors

    with ThreadPoolExecutor(max_workers=worker) as executor:
        # Execute sensor extraction for each of the listed station in list_needed_stations
        # Extraction of said ids would be done in a singular thread, where the number of concurrent extractions done in one thread is set to 6, allowing details of
        # 6 stations to be processed at one given time
        results = list(executor.map(extract_sensor_ids, [v[1] for v in list(list_needed_stations.items())], [v[0] for v in list(list_needed_stations.items())]))

        for res in results:
            # stores the execution results in the station_details_full dictionary by its station name
            station_details_full[res['name']] = res

    return station_details_full # returns a dictionary of station names & their attributes

# Function to convert station details dictionary to csv format
def convert_station_details_csv(station_details: dict, folder_name='datasets_raw'):
    # Initialization of datasets directory for raw csv datasets
    path_datasets_raw = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(path_datasets_raw):
        os.makedirs(path_datasets_raw)
        
    sensors_df_draft = {
        'station_id': [],
        'station': [],
        'long': [],
        'lat': [],
        'sensor_id': []
    }

    for station in station_details:
        # Iterates through each station and sensors, storing said data into a list within the sensors_df_draft dictionary, which would be used to create a pandas
        # dataframe, which would then be used to conver the data into clean csv
        curr_station = station_details[station]
        for sensor in station_details[station]['sensors']:
            sensors_df_draft['station_id'].append(curr_station['id'])
            sensors_df_draft['station'].append(curr_station['name'])
            sensors_df_draft['long'].append(curr_station['long'])
            sensors_df_draft['lat'].append(curr_station['lat'])
            sensors_df_draft['sensor_id'].append(sensor)

    station_sensors_df = pd.DataFrame(sensors_df_draft)
    station_sensors_df.to_csv(os.path.join(path_datasets_raw, 'station_sensors.csv'), index=False) # conversion to csv, stored in the datasets_raw folder
    print(f'Saved station details to: {os.path.join(path_datasets_raw, 'station_sensors.csv')}\n')

    return station_sensors_df # returns the stations details dataframe


# ################################################ STATION PM2.5 READINGS EXTRACTION FUNCTIONS ################################################

# Function to verify local thread
def get_session(headers=headers):
    if not hasattr(thread_local, "session"):
        # Checks if the local thread has created a session, which would allow threading. If the thread is new or have not sent a request,
        # a new session is created for a specific thread, with the specific header that has been set earlier
        thread_local.session = requests.Session()
        thread_local.session.headers.update(headers)
    return thread_local.session # returns the created session

# Function to get chunks of readings
def get_hourly_chunk(sensor_id, station: str, start: str, end: str, limit=1000, headers=headers):
    page=1 # sets it so that the first chunk (page) is taken
    session = get_session() # retrives current session
    length_processed=0 # stores the amount of extracted readings

    print(f'Processing {station}: sensor {sensor_id}\n')

    # sets a temporary end date for only one year of readings to prevent the openAQ API to be overwhelmed
    if int(start[:4])==int(end[:4]):
        end_temp = end
    else:
        end_temp = datetime.datetime.strptime(start, '%Y-%m-%d')+dateutil.relativedelta.relativedelta(years=1)
    timestamp_from, timestamp_to, values = [], [], [] # stores the PM2.5 readings and its timestamps
    try:
        while True:
            while True:
                end_curr=str(end_temp).split(' ')[0] # takes the date (i.e. from 2023-04-10 00:00:00, only 2023-04-10 is taken)
                params = { # parameters for the API call
                    'datetime_from': start,
                    'datetime_to': end_curr,
                    'limit':limit,
                    'page':page
                }

                try:
                    # API call is executed
                    hourly_test = session.get(f'https://api.openaq.org/v3/sensors/{sensor_id}/hours', headers=headers, params=params)
                    print(f'{hourly_test} -- processing page {page} (end date {end_curr} - {station}: sensor {sensor_id})')
                    
                    # In case the request temporarily blocked, the loop is stopped for one minute (to reset per minute API call limit),
                    # and then the loop is resumed, allowing the request to be retried
                    if hourly_test.status_code in (408, 500, 429):
                        time.sleep(60)
                        continue
                    hourly_test.raise_for_status()

                    for val in hourly_test.json()['results']:
                        # Loops through, exracts and stores information regarding PM2.5 readings and its timestamp (in UTC) from request result
                        time_from=val['period']['datetimeFrom']['utc']
                        time_to=val['period']['datetimeTo']['utc']
                        value=val['value']

                        timestamp_from.append(time_from)
                        timestamp_to.append(time_to)
                        values.append(value)
                        #print(f'{value} - at {time_}')

                    length_processed+=len(hourly_test.json()['results']) # adds the amount of records processed
                    print(f'Processed {length_processed} records')

                    time.sleep(1) # keeps slight delay to prevent endpoint from being overwhelmed

                    if len(hourly_test.json()['results'])<params['limit']:
                        # Case when the length of results are less than the limit (meaning either data is insufficient or all of the data has been extracted)
                        # the inner loop is stopped, and continues to the next chunk of data
                        print(f'Finished processing all records - {station}: sensor {sensor_id}')
                        break
                    page+=1 # page is added by 1 increment for processing the next chunk
                except Exception as e:
                    # If exception occurs within inner loop (i.e. due to request rejection), the loop is re-attempted
                    print(f'{e}\nError within extract loop')
                    continue
            
            if end_temp==end or length_processed==0:
                # If the end date is reached, or no data is processed after one loop, then the whole loop is stopped
                break
            
            time.sleep(60) # 1-minute delay to prevent endpoint from being overwhelmed
            page=1 # resetting page to 1 after one full chunk is processed
            if end_temp.year==int(end[:4]):
                # when the current chunk's end date has already reached the year 2026, the new temporary end date is set to the actual end date (as set in function parameters)
                start = str(end_temp).split(' ')[0]
                end_temp = end
                continue
            start = str(end_temp).split(' ')[0] # renewing start date for each chunk of data 
            end_temp = datetime.datetime.strptime(start, '%Y-%m-%d')+dateutil.relativedelta.relativedelta(years=1) # renewing end date for each chunk of data 
            print(' ')

        return values, timestamp_from, timestamp_to

    except Exception as e:
        # If exception occurs within the outer loop, the whole loop is stopped
        print(e)
        return

# Function for executing PM2.5 readings extraction
def get_pm25_reads(list_needed_stations: dict, start: str, end:str, limit=1000, header=headers, num_workers=9, folder_name='datasets_raw'):
    station_details = get_station_details(list_needed_stations) # extracts details of each station
    convert_station_details_csv(station_details, folder_name=folder_name) # saves each station's details to csv format
    print(' ')

    sensor_reads = {} # will store each sensor's Pm2.5 readings dataset
    sensors = [] # will store sensor id, station id, and station name for later
    for loc, content in station_details.items():
        for sensor in content['sensors']:
            sensors.append((loc, content['id'], sensor))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Execute the extraction process on the allocated thread, which is done on 9 worker nodes for faster processing (allowing 9 individual sensors' readings to be extracted at
        # one given time), where PM2.5 readings extraction is done on each individual sensor
        results = {executor.submit(get_hourly_chunk, sensor, loc_, start, end, limit, header): (loc_, loc_id, sensor) for loc_, loc_id, sensor in sensors}

        for res in as_completed(results):
            # The resulting dictionary of results is looped through, to extract Pm2.5 readings and its timestamp, as well as to insert station and sensor
            # information within the returned dataset
            loc_, loc_id, sensor = results[res]
            try:
                values, timestamp_from, timestamp_to = res.result()
                # Creates dataframe from the extracted data
                sensor_reads[sensor] = pd.DataFrame({
                    'location_id': [loc_id]*len(values), # stores station id
                    'location_name': [loc_]*len(values), # stores station name
                    'sensor_id': [sensor]*len(values), # stores sensor id
                    'values_pm25': values, # stores PM2.5 readings
                    'timestamp_from': timestamp_from, # stores time when reading starts
                    'timestamp_to': timestamp_to, # stores time when reading stops
                })
            except Exception as e:
                # When error occurs the entire process is to be restarted
                print(e)
                return
    
    # Initialization of datasets directory for raw csv datasets
    path_datasets_raw = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(path_datasets_raw):
        os.makedirs(path_datasets_raw)

    # Combines data from all sensors and save said data into csv format
    df_pm25_readings = pd.concat([sensor_reads[i] for i in sensor_reads.keys()], axis=0, ignore_index=True)
    df_pm25_readings.to_csv(os.path.join(path_datasets_raw, 'sensor_readings.csv'), index=False)
    print(f"\nSaved readings to: {os.path.join(path_datasets_raw, 'sensor_readings.csv')}")

    return df_pm25_readings # returns dataframe of PM 2.5 readings


# ################################################ STATION WEATHER CONDITION EXTRACTION FUNCTIONS ################################################

# Function for getting a station's weather readings
def get_weather_readings_station(station: str, station_id, long: float, lat:float, start: str, end: str):
    try:
        # API call
        weather = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start}&end_date={end}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,rain&timezone=auto')
        print(f'\nProcessing {station} - long {long} & lat {lat}, status code: {weather.status_code}')
        weather.raise_for_status() # if the endpoint rejects request, an exception is raised
        
        dict_result = weather.json()['hourly'] # extracting API call result, converting it to dictionary and taking only the attribute which contains hourly readings
        dict_result['station_id_london'] = station_id # adds station id to the dictionary
        dict_result['station_name'] = station # adds station name to the dictionary
        dict_result['long'] = long # adds station longitude to the dictionary
        dict_result['lat'] = lat # adds station latitude to the dictionary

        df_weather = pd.DataFrame(dict_result) # converts the dictionary to pandas dataframe
        print(f'Finished on {station}')
        return df_weather # returns the dataframe
    
    except Exception as e:
        # If an exception is raised, the extraction will be re-attempted after 30 seconds
        print(f'Fail on {station} due to {e}\nRETRYING...')
        time.sleep(30)
        return get_weather_readings_station(station, station_id, long, lat, start, end)

# Function for executing extraction of weather readings for each station
def get_weather_readings(start: str, end: str, list_needed_stations: dict, workers=6, folder_name='datasets_raw'):
    station_details = get_station_details(list_needed_stations) # getting the details of each station (i.e. longitude, latitude, and id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Executes extraction on the allocated thread, with the max workers set to 6 as default, allowing six of the station's weather readings to be extracted at one given time
        station_names = list(station_details.keys()) # extracts station names
        station_ids = [station_details[station]['id'] for station in station_names] # extracts station ids
        longitudes = [station_details[station]['long'] for station in station_names] # extracts the longitude of a station
        latitudes = [station_details[station]['lat'] for station in station_names] # extracts the latitude of a station
        items = list(executor.map(get_weather_readings_station, station_names, station_ids, longitudes, latitudes, itertools.repeat(start), itertools.repeat(end))) # executes extraction

    # Initialization of datasets directory for raw csv datasets
    path_datasets_raw = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(path_datasets_raw):
        os.makedirs(path_datasets_raw)

    # Combines all extraction results (which is in the form of pandas dataframe) and stores the combined weather readings to the path_datasets_raw directory in csv format
    df_weathers = pd.concat(items, axis=0, ignore_index=True)
    df_weathers.to_csv(os.path.join(path_datasets_raw, 'weathers_dataset.csv'), index=False)
    return df_weathers # returns the combined weather readings dataframe 

# ################################################ UP-TO-DATE READINGS EXTRACTION ################################################

# Function for Extracting Up-To-Date PM2.5 Readings
def get_pm25_reads_short(station_details_recent: dict, header: dict, limit=1000, num_workers=9):
    sensor_reads = {} # will store each sensor's Pm2.5 readings dataset
    sensors = [] # will store sensor id, station id, and station name for later
    for loc, content in station_details_recent.items():
        for sensor in content['sensors']:
            sensors.append((loc, content['id'], sensor, content['start'], content['end']))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Execute the extraction process on the allocated thread, which is done on 9 worker nodes for faster processing (allowing 9 individual sensors' readings to be extracted at
        # one given time), where PM2.5 readings extraction is done on each individual sensor
        results = {executor.submit(get_hourly_chunk, sensor, loc_, start, end, limit, header): (loc_, loc_id, sensor, start, end) for loc_, loc_id, sensor, start, end in sensors}

        for res in as_completed(results):
            # The resulting dictionary of results is looped through, to extract Pm2.5 readings and its timestamp, as well as to insert station and sensor
            # information within the returned dataset
            loc_, loc_id, sensor, start, end = results[res]
            try:
                values, timestamp_from, timestamp_to = res.result()
                # Creates dataframe from the extracted data
                sensor_reads[sensor] = pd.DataFrame({
                    'location_id': [int(loc_id)]*len(values), # stores station id
                    'location_name': [loc_]*len(values), # stores station name
                    'sensor_id': [int(sensor)]*len(values), # stores sensor id
                    'values_pm25': values, # stores PM2.5 readings
                    'timestamp_from': timestamp_from, # stores time when reading starts
                    'timestamp_to': timestamp_to, # stores time when reading stops
                })
            except Exception as e:
                # When error occurs the entire process is to be restarted
                print(e)
                
    # Combines data from all sensors
    df_pm25_readings = pd.concat([sensor_reads[i] for i in sensor_reads.keys()], axis=0, ignore_index=True)

    return df_pm25_readings 

# Function for executing extraction of Up-to-Date Weather Readings
def get_weather_readings_short(station_details_recent: dict, workers=6):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Executes extraction on the allocated thread, with the max workers set to 6 as default, allowing six of the station's weather readings to be extracted at one given time
        station_names = list(station_details_recent.keys()) # extracts station names
        station_ids = [station_details_recent[station]['id'] for station in station_names] # extracts station ids
        longitudes = [station_details_recent[station]['long'] for station in station_names] # extracts the longitude of a station
        latitudes = [station_details_recent[station]['lat'] for station in station_names] # extracts the latitude of a station
        start = [station_details_recent[station]['start'].split('T')[0] for station in station_names] # extracts start date
        end = [station_details_recent[station]['end'].split('T')[0] for station in station_names] # extracts end date
        items = list(executor.map(get_weather_readings_station, station_names, station_ids, longitudes, latitudes, start, end)) # executes extraction

    # Combines all extraction results (which is in the form of pandas dataframe)
    df_weathers = pd.concat(items, axis=0, ignore_index=True)
    return df_weathers # returns the combined weather readings dataframe 