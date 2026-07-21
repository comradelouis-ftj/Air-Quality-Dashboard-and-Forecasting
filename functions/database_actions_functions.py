import pandas as pd
import psycopg2
from psycopg2 import sql

import datetime

from functions.ingestion_functions import get_pm25_reads, get_weather_readings, get_pm25_reads_short, get_weather_readings_short
from functions.transformation_functions import normalize_save_pm25_weather_readings, normalize_save_pm25_weather_readings_short

import os
import shutil

# ####################################################### INITIALIZATION #######################################################
# Function for Creating Schema & Table (if does not exist)
def create_tables(cursor, connection):
    query = '''
        CREATE SCHEMA IF NOT EXISTS staging;

        CREATE TABLE IF NOT EXISTS staging.station_details (
            station_id INT PRIMARY KEY,
            station_name VARCHAR,
            longitude FLOAT,
            latitude FLOAT
        );

        CREATE TABLE IF NOT EXISTS staging.station_sensors (
            sensor_id INT PRIMARY KEY,
            station_id INT,
            CONSTRAINT fk_sensor_station FOREIGN KEY (station_id) REFERENCES staging.station_details(station_id)
            ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS staging.weather_readings (
            weather_id VARCHAR PRIMARY KEY,
            sensor_id INT,
            time_reading TIMESTAMP,
            temperature_2m FLOAT,
            relative_humidity_2m FLOAT,
            wind_speed_10m FLOAT,
            surface_pressure FLOAT,
            rain FLOAT,
            pm25 FLOAT,
            CONSTRAINT fk_weather_sensor FOREIGN KEY (sensor_id) REFERENCES staging.station_sensors(sensor_id)
            ON DELETE CASCADE
        );
    '''
    cursor.execute(query)
    connection.commit()
    #print(cursor.fetchall())

# Database Intialization Function
def initialize_db(username: str, password: str, name_db: str):
    # Initializing connection
    connect = psycopg2.connect(f'host=localhost dbname=postgres user={username} password={password}')
    connect.autocommit=True # autocommit set to True, in case database does not exist

    with connect.cursor() as cursor:
        query="SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_database WHERE datname=%s);" # checks if database exists
        cursor.execute(query, (name_db,))
        res = cursor.fetchone()[0]

        print(f'Existence of DB {name_db}: {res}\n')

        if res:
            # If database exists, the previous connection is closed, and a new one created. In case it does not have the schema or certain tables, 
            # the create_tables function is called to create the schema and tables. If the tables & schema already exists, then PostgreSQL will
            # simply skip the queries within the create_tables function
            connect.close()
            connect_n = psycopg2.connect(f'host=localhost dbname={name_db} user={username} password={password}')
            cursor_n = connect_n.cursor()

            create_tables(cursor_n, connect_n)
        
            return connect_n, cursor_n
        
        else:
            # If database does not exist, then it is created via the below DDL code
            print(f'Making DB {name_db}')
            cursor.execute(sql.SQL('CREATE DATABASE {db};').format(db=sql.Identifier(name_db)))
            connect.close()
            
            return initialize_db(username, password, name_db) # calls back the function, so the tables could be inserted

# ####################################################### BULK INSERT #######################################################
# Function for Inserting in Bulk
def bulk_insert_values_to_db(username: str, password: str, name_db: str, folder_name='datasets_clean'):
    connection, cursor = initialize_db(username, password, name_db) # calls the initialize_db function to get cursor & connection to database

    # Extracts paths to cleaned csv files
    path_clean_dataset = os.path.join(os.getcwd(), folder_name)
    sensor_detail_dir = os.path.join(path_clean_dataset, 'sensor_stationid_pair.csv')
    station_detail_dir = os.path.join(path_clean_dataset, 'station_details.csv')
    weather_reads_dir = os.path.join(path_clean_dataset, 'weather_pm25_readings.csv')

    # The list of dictionaries below will be used for the bulk inserts, which will be done sequentially
    dict_tables = [
        {
            'table': 'staging.station_details', # stores name of table in database
            'columns': '(station_id, station_name, longitude, latitude)', # stores database columns
            'dir': station_detail_dir # stores directory of the specific csv file for this table
        },
        {
            'table': 'staging.station_sensors',
            'columns': '(sensor_id, station_id)',
            'dir': sensor_detail_dir
        },
        {
            'table': 'staging.weather_readings',
            'columns': '(weather_id, sensor_id, time_reading, temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure, rain, pm25)',
            'dir': weather_reads_dir
        }
    ]

    # Truncates existing tables first. Note that this is done since the assumption is, this function will be used for massive bulk insert at the
    # very start, and as such, would only be done only when all data has become redundant or only before preparation
    cursor.execute('TRUNCATE TABLE staging.weather_readings, staging.station_sensors, staging.station_details')
    print('Tables truncated')
    connection.commit()

    for tbl in dict_tables:
        print(f'Processing for {tbl['table']}')
        # loops through all the dictionaries
        with open(tbl['dir'], 'r', encoding='utf-8') as f:
            # Creates the COPY SQL script to copy contents of the csv file into the PostgreSQL database
            script =f"""
                COPY {tbl['table']} {tbl['columns']} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');
            """
            cursor.copy_expert(sql=script, file=f)
        print(f'Completed for {tbl['table']}')
    connection.commit() # commits SQL script

# ####################################################### CREATE & EXTRACT VIEW #######################################################
# Function for Creating/Updating View
def create_view(connection, cursor):
    cursor.execute('''
        CREATE SCHEMA IF NOT EXISTS aggr;

        CREATE OR REPLACE VIEW aggr.weather_readings_mlready AS
        SELECT 
            wr.weather_id, wr.sensor_id, sd.station_id, sd.station_name, wr.time_reading,
            wr.temperature_2m, wr.relative_humidity_2m, wr.wind_speed_10m, wr.surface_pressure, wr.rain, wr.pm25, sd.longitude, sd.latitude 
        FROM staging.weather_readings wr
        LEFT JOIN staging.station_sensors ss ON wr.sensor_id=ss.sensor_id
        LEFT JOIN staging.station_details sd ON ss.station_id=sd.station_id;
    ''')
    connection.commit()

# Function for Storing Full Aggregated Data
def serve_whole_data(username_postgres: str, pw_postgres: str, db_name: str, limit=1000):
    connection, cursor = initialize_db(username_postgres, pw_postgres, db_name) # database initialization & connection
    table_results = [] # will store all dataframe results

    # Creates aggregated dataset path for the cleaned, aggregated data
    path_aggr_dataset = os.path.join(os.getcwd(), 'datasets_aggr')
    if not os.path.exists(path_aggr_dataset):
        os.makedirs(path_aggr_dataset)

    create_view(connection, cursor) # creates/updates view
    
    # Loop for extracting data from view, the process is done in chunks in order to prevent overloading of computation resources
    offset=0
    while True:
        # Query for extracting the data from viewin chunks
        table_result = pd.io.sql.read_sql_query(f'''
            SELECT * FROM aggr.weather_readings_mlready
            LIMIT {limit}
            OFFSET {offset};
        ''', connection)
        table_results.append(table_result) # stores the dataframe

        offset+=limit # offset by x amount, will keep increasing to accomodate the limit
        if len(table_result)<limit: # when the returned dataframe's length is under the limit, the loop breaks
            break
    
    # Combines all returned dataframes and stores it into csv format for easy retrieval
    result_concat = pd.concat(table_results, axis=0, ignore_index=True)
    result_concat.to_csv(os.path.join(path_aggr_dataset, 'weather_readings_mlready.csv'), index=False)
    print(f'Saved cleaned, read-ML data to {os.path.join(path_aggr_dataset, 'weather_readings_mlready.csv')}')

    connection.close() # close database connection
    return result_concat # returns the dataframe

# ####################################################### EXTRACT UP-TO-DATE DATA #######################################################
# Function for Loading Up-to-Date Readings to Database
def load_to_db(connection, cursor, df_cleaned):
    # Looping through all dataframe records and inserting it into the database (note: this method is chosen since it is not expected that each inserts would involve
    # as much records as a full csv-tp-database insert, instead this type of insert should only handle at most a few hundred records)
    for val in df_cleaned.values:
        cursor.execute('''
            INSERT INTO staging.weather_readings (weather_id, sensor_id, time_reading, temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure, rain, pm25)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (weather_id)
            DO UPDATE SET 
                sensor_id = EXCLUDED.sensor_id, 
                time_reading = EXCLUDED.time_reading, 
                temperature_2m = EXCLUDED.temperature_2m, 
                relative_humidity_2m = EXCLUDED.relative_humidity_2m, 
                wind_speed_10m = EXCLUDED.wind_speed_10m, 
                surface_pressure = EXCLUDED.surface_pressure, 
                rain = EXCLUDED.rain, 
                pm25 = EXCLUDED.pm25;
        ''', [v for v in list(val)])
        connection.commit() # commit changes

# Function for Extracting Weather & PM2.5 Readings Up Until Current Date and Time
# note: this function is purely for adding readings that comes from stations/sensors that are already recorded in the database, and as such, will not be able to 
#       ingest readings coming from new sensors/stations
def extract_readinga_to_current_date(username_postgres: str, pw_postgres: str, db_name: str, headers: dict):
    time_now = datetime.datetime.now().strftime('%Y-%m-%dT%H:00:00') # extracting the current day's timestamp, to ensure more records are available # - datetime.timedelta(days=1)
    print(time_now)
    limit_old_timestamp = None # will store the last timestamp recorded in database tables
    station_start_ends = {} # will store each station's details (i.e. coordinates, id, name, and sensors)

    connection, cursor = initialize_db(username_postgres, pw_postgres, db_name) # connecting to local database
    cursor.execute('SELECT station_name, station_id FROM staging.station_details;')

    # Creates list of stations
    list_needed_stations = {}
    for station, id_station in cursor.fetchall():
        list_needed_stations[station] = id_station
    
    for station in list_needed_stations:
        # Loops through the stations and its ids

        station_start_ends[station] = { # stores the station's details
            'id': list_needed_stations[station],
            'sensors': [],
            'end': time_now,
            'start': None
        }

        # Getting a list of sensors for a specific station
        cursor.execute("""
            SELECT sensor_id FROM (
                SELECT * FROM staging.station_sensors WHERE station_id=%s
            ) AS temp;
        """, [list_needed_stations[station],])
        res_sensor = [v[0] for v in cursor.fetchall()]
        
        # Extracting coodinates for a specific station
        cursor.execute('SELECT longitude, latitude FROM staging.station_details WHERE station_id=%s;', [290183,])
        res_coor = cursor.fetchall()[0]
        station_start_ends[station]['long'],  station_start_ends[station]['lat'] = res_coor[0], res_coor[1]

        # Retrieving maximum end date for a station, done based on a sensor's reading, where the sensor with the most recent reading's most recent timestamp will be
        # chosen as the end date for the specific station
        for sensor in res_sensor:
            station_start_ends[station]['sensors'].append(sensor)
            cursor.execute("""
                SELECT MAX(time_reading) AS most_recent_read FROM (
                    SELECT wr.sensor_id, ss.station_id, wr.time_reading
                    FROM staging.weather_readings wr
                    LEFT JOIN staging.station_sensors ss ON wr.sensor_id=ss.sensor_id
                    WHERE wr.sensor_id=%s
                ) AS temp;
            """, [sensor,])
            res = cursor.fetchone()[0]
            print(f'Max datetime for {station}: {res}')

            # Updating the start date
            if res != None and (station_start_ends[station]['start'] == None or res > datetime.datetime.strptime(station_start_ends[station]['start'], '%Y-%m-%dT%H:%M:%S')):
                station_start_ends[station]['start'] = res.strftime('%Y-%m-%dT%H:00:00')

            if res is not None:
                # Checking if the data within database is already the newest iteration (either difference between newest date and the current time is only one hour, or there is no difference 
                # between current time and the newest record's timestamp)
                if (datetime.datetime.strptime(time_now.replace('T', ' '), '%Y-%m-%d %H:00:00')-res).total_seconds()==3600 or datetime.datetime.strptime(time_now.replace('T', ' '), '%Y-%m-%d %H:00:00')==res:
                    print('Timestamp too new, try again later')
                    return None

            # Updating the latest timestamp of readings within database
            if limit_old_timestamp == None and res != None:
                limit_old_timestamp = res.strftime('%Y-%m-%dT%H:00:00')
            elif limit_old_timestamp != None and res != None:
                if res < datetime.datetime.strptime(limit_old_timestamp.replace('T', ' '), '%Y-%m-%d %H:%M:%S'):
                    limit_old_timestamp = res.strftime('%Y-%m-%dT%H:00:00')
    
    # Extracting and cleaning weather and PM2.5 readings using other functions
    readings_pm25 = get_pm25_reads_short(station_start_ends, headers) # extracts PM2.5
    readings_weather = get_weather_readings_short(station_start_ends) # extracts weather readings
    if len(readings_pm25)==0 or len(readings_weather)==0:
        # If either PM2.5 or weather readings did not return anything then nothing is returned
        print('No newer records found, API endpoint may be down, try again later...')
        return None

    merged_readings = normalize_save_pm25_weather_readings_short(readings_pm25, readings_weather) # cleans and merged PM2.5 and weather readings
    #print(merged_readings)

    # Cutting parts of the dataframe, ensuring that datetimes exceeding the current datetime is excluded from the final dataframe
    time_now_dt = datetime.datetime.strptime(time_now.replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    limit_old_timestamp_dt = datetime.datetime.strptime(limit_old_timestamp.replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    merged_readings = merged_readings[(merged_readings['time']<time_now_dt) & (merged_readings['time']>=limit_old_timestamp_dt)]
    merged_readings.reset_index(inplace=True, drop=True)
    #print(merged_readings)

    load_to_db(connection, cursor, merged_readings)

    connection.close() # closing database connection

# ####################################################### EXTRACT DIFFERENT STATIONS' READINGS #######################################################
# Function for Inserting Readings from New Station in Bulk (from CSV file)
def bulk_insert_new_station(username: str, password: str, name_db: str, folder_name:str):
    connection, cursor = initialize_db(username, password, name_db) # calls the initialize_db function to get cursor & connection to database

    # Extracts paths to cleaned csv files
    path_clean_dataset = os.path.join(os.getcwd(), folder_name)
    sensor_detail_dir = os.path.join(path_clean_dataset, 'sensor_stationid_pair.csv')
    station_detail_dir = os.path.join(path_clean_dataset, 'station_details.csv')
    weather_reads_dir = os.path.join(path_clean_dataset, 'weather_pm25_readings.csv')

    # The list of dictionaries below will be used for the bulk inserts, which will be done sequentially
    dict_tables = [
        {
            'name': 'station_details',
            'columns': ['station_id', 'station_name', 'longitude', 'latitude'], # stores database columns
            'dir': station_detail_dir, # stores directory of the specific csv file for this table
            'pk': 'station_id', # stores primary key of the specific table
        },
        {
            'name': 'station_sensors',
            'columns': ['sensor_id', 'station_id'],
            'dir': sensor_detail_dir,
            'pk': 'sensor_id',
            'update': 'station_id=EXCLUDED.station_id'
        },
        {
            'name': 'weather_readings',
            'columns': ['weather_id', 'sensor_id', 'time_reading', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure', 'rain', 'pm25'],
            'dir': weather_reads_dir,
            'pk': 'weather_id'
        }
    ]

    for tbl in dict_tables:
        print(f'Processing for {tbl['name']}')
        # Loops through all the dictionaries' csv files and appends its values into the database
        with open(tbl['dir'], 'r', encoding='utf-8') as f:
            # Creates temporrary sql table (will be removed after connection closes) which temporarily stores the data from the csv files
            cursor.execute(sql.SQL('CREATE TEMP TABLE {name} (LIKE staging.{name});').format(name=sql.Identifier(tbl['name']))) 

            # Converts the list of columns into sql-identifiable syntax
            columns = [sql.Identifier(col.strip()) for col in tbl['columns']]
            cols_formatted = sql.SQL(', ').join(columns)

            # Creates syntax for handling cases where an id is duplicated (example syntax: station_name=EXCLUDED.station_name), which will be used later 
            # in the DO UPDATE SET ... syntax
            updates = ', '.join([f'{col}=EXCLUDED.{col}' for col in tbl['columns']])
            updates_sql = sql.SQL(updates)

            # Copies contents of a csv file and stores it in a temporary table first
            script = f"COPY {tbl['name']} ({', '.join(tbl['columns'])}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');"
            cursor.copy_expert(sql=script, file=f)

            # Inserts contents of the temporary table into the actual tables within the database. Here, ON CONFLICT & DO UPDATE SET ... essentially
            # is a system to handle duplicate primary keys, where in case a duplicate exists, the current key's record  in the database table will 
            # be replaced with the new set of records within this loop. If there is no duplicating key, a standard insert will be performed
            cursor.execute(sql.SQL('''
                INSERT INTO staging.{name} ({columns})
                SELECT {columns} FROM {name}
                ON CONFLICT ({pk})
                DO UPDATE SET {update};
            ''').format(
                name=sql.Identifier(tbl['name']),
                columns=cols_formatted,
                pk=sql.Identifier(tbl['pk']),
                update=updates_sql # note: this uses SQL, as the identifier can be replaced by whole SQL instead
            ))

            connection.commit() # commits the changes
        print(f'Finished on {tbl['name']}\n')
    
    connection.close() # close database connection

# Function for Inserting New Station in Database
def insert_new_station_readings(username_postgres: str, pw_postgres: str, db_name: str, headers, station_names, station_ids, folder_name='datasets_temp', remove_folder_at_end=False):
    list_needed_stations={} # stores the stations

    if type(station_names)==list:
        # When a list of stations is inserted, the all stations are looped through and stored within a dictionary
        for i in range(len(station_names)):
            list_needed_stations[station_names[i]] = station_ids[i]
    else:
        # Otherwise, the singular station is stored in the dictionary
        list_needed_stations[station_names]=station_ids

    connection, cursor = initialize_db(username_postgres, pw_postgres, db_name) # initialize database connection
    cursor.execute('SELECT MAX(time_reading) FROM staging.weather_readings;')
    end_time = cursor.fetchone()[0]
    end = end_time.strftime('%Y-%m-%d')

    cursor.execute('SELECT MIN(time_reading) FROM staging.weather_readings;')
    start_time = cursor.fetchone()[0]
    start = start_time.strftime('%Y-%m-%d')

    print(f'Start: {start_time}, end: {end_time}')

    cursor.execute('SELECT DISTINCT station_id FROM staging.station_sensors;') # extracts all existing station ids in the database
    list_station_ids = [s[0] for s in cursor.fetchall()] # stores the extracted station ids in a list

    # Loops theough the station in the list_needed_stations dictionary, to find and remove stations which already existed in the database
    stations_already_exist = []
    for station, id_station in list_needed_stations.items():
        if id_station in list_station_ids:
            stations_already_exist.append(station)
    for station_name in stations_already_exist:
        list_needed_stations.pop(station_name)
        print(f'Deleting {station_name} because it already exists....')

    if len(list_needed_stations)==0: # if all stations already exists in the database, the function returns nothing
        print('Empty')
        return

    _ = get_pm25_reads(list_needed_stations, start, end, header=headers, folder_name=folder_name) # extracting and storing PM2.5 readings
    _ = get_weather_readings(start, end, list_needed_stations, folder_name=folder_name) # extracting and storing weather readings
    id_station_detail, merged_weather_pm25, station_details = normalize_save_pm25_weather_readings(folder_name_raw=folder_name, folder_name_clean=folder_name) # cleaning and storing weather readings
    merged_weather_pm25 = merged_weather_pm25[(merged_weather_pm25['time']<=end_time) & (merged_weather_pm25['time']>start_time)]
    path_clean_dataset = os.path.join(os.getcwd(), folder_name)
    merged_weather_pm25.to_csv(os.path.join(path_clean_dataset, 'weather_pm25_readings.csv'), index=False)

    bulk_insert_new_station(username_postgres, pw_postgres, db_name, folder_name=folder_name) # inserting weather readings from the new station(s) into the database

    if remove_folder_at_end: # if removing the temporary folder is set to true, then it will be removed along with its contents
        shutil.rmtree(path_clean_dataset)
        print(f'Removed temporary folder at {path_clean_dataset}')
    connection.close() # close connection

    return id_station_detail, merged_weather_pm25, station_details # returns the cleaned dataframes