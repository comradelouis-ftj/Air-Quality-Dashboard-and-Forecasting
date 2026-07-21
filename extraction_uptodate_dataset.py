from functions.database_actions_functions import extract_readinga_to_current_date, insert_new_station_readings, serve_whole_data

import threading

import os
from dotenv import load_dotenv

if __name__=='__main__':
    load_dotenv() # load environment variables
    thread_local = threading.local() # Initializing thread local storage

    list_tables = ['station_details', 'station_sensors', 'weather_readings']
    username_postgres = os.getenv('USER_POSTGRES')
    pw_postgres = os.getenv('PW_POSTGRES')
    db_name = 'data_warehouse_weather'

    headers = {
        'X-API-Key': os.getenv('API_OPENAQ') # initializing request header
    }

    merged_readings=extract_readinga_to_current_date(username_postgres, pw_postgres, db_name, headers)

    df = serve_whole_data(username_postgres, pw_postgres, db_name, limit=10000)