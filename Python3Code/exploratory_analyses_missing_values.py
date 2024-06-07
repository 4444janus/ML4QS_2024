import pandas as pd
import numpy as np
from datetime import datetime, timezone
from crowdsignals_ch3_outliers import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection
from util.VisualizeDataset import VisualizeDataset
# Data samenvoegen
# De tijd bij elkaar optellen, dus voor de 2e heb je de huidige tijd per datapunt + eind tijd van 1e

def get_combined_data():
    data_1 = pd.read_csv('datasets/own_data/auto_10hz/formatted_car_accelerometer.csv')
    data_2 = pd.read_csv('datasets/own_data/auto_10hz_v2/formatted_car_accelerometer.csv')
    data_3 = pd.read_csv('datasets/own_data/auto_10hz_v3/formatted_car_accelerometer.csv')
    data_4 = pd.read_csv('datasets/own_data/fietsen_10hz/formatted_bike_accelerometer.csv')
    data_5 = pd.read_csv('datasets/own_data/lopen_10hz_v1/formatted_walk_accelerometer.csv')
    data_6 = pd.read_csv('datasets/own_data/lopen_10hz_v2/formatted_walk_accelerometer.csv')
    data_7 = pd.read_csv('datasets/own_data/lopen_10hz_v3/formatted_walk_accelerometer.csv')
    data_8 = pd.read_csv('datasets/own_data/tram_10hz_v1/formatted_tram_accelerometer.csv')

    data_gyro_1 = pd.read_csv('datasets/own_data/auto_10hz/formatted_car_gyroscope.csv')
    data_gyro_2 = pd.read_csv('datasets/own_data/auto_10hz_v2/formatted_car_gyroscope.csv')
    data_gyro_3 = pd.read_csv('datasets/own_data/auto_10hz_v3/formatted_car_gyroscope.csv')
    data_gyro_4 = pd.read_csv('datasets/own_data/fietsen_10hz/formatted_bike_gyroscope.csv')
    data_gyro_5 = pd.read_csv('datasets/own_data/lopen_10hz_v1/formatted_walk_gyroscope.csv')
    data_gyro_6 = pd.read_csv('datasets/own_data/lopen_10hz_v2/formatted_walk_gyroscope.csv')
    data_gyro_7 = pd.read_csv('datasets/own_data/lopen_10hz_v3/formatted_walk_gyroscope.csv')
    data_gyro_8 = pd.read_csv('datasets/own_data/tram_10hz_v1/formatted_tram_gyroscope.csv')

    dt = datetime(2024, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
    timestamp = dt.timestamp()
    to_subtract = int(timestamp)

    # Make the time stamps for all the data continuous in time. Data 2 should start where data 1 ends, etc.
    # Take into account that timestamps is in unix time, so only add the difference between the last timestamp of the previous data and the first timestamp of the current data
    data_2['timestamps'] = data_2['timestamps'] + data_1['timestamps'].iloc[-1] - to_subtract
    data_3['timestamps'] = data_3['timestamps'] + data_2['timestamps'].iloc[-1] - to_subtract
    data_4['timestamps'] = data_4['timestamps'] + data_3['timestamps'].iloc[-1] - to_subtract
    data_5['timestamps'] = data_5['timestamps'] + data_4['timestamps'].iloc[-1] - to_subtract
    data_6['timestamps'] = data_6['timestamps'] + data_5['timestamps'].iloc[-1] - to_subtract
    data_7['timestamps'] = data_7['timestamps'] + data_6['timestamps'].iloc[-1] - to_subtract
    data_8['timestamps'] = data_8['timestamps'] + data_7['timestamps'].iloc[-1] - to_subtract

    data_gyro_2['timestamps'] = data_gyro_2['timestamps'] + data_gyro_1['timestamps'].iloc[-1] - to_subtract
    data_gyro_3['timestamps'] = data_gyro_3['timestamps'] + data_gyro_2['timestamps'].iloc[-1] - to_subtract
    data_gyro_4['timestamps'] = data_gyro_4['timestamps'] + data_gyro_3['timestamps'].iloc[-1] - to_subtract
    data_gyro_5['timestamps'] = data_gyro_5['timestamps'] + data_gyro_4['timestamps'].iloc[-1] - to_subtract
    data_gyro_6['timestamps'] = data_gyro_6['timestamps'] + data_gyro_5['timestamps'].iloc[-1] - to_subtract
    data_gyro_7['timestamps'] = data_gyro_7['timestamps'] + data_gyro_6['timestamps'].iloc[-1] - to_subtract
    data_gyro_8['timestamps'] = data_gyro_8['timestamps'] + data_gyro_7['timestamps'].iloc[-1] - to_subtract


    # Combine the data
    data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8])
    data_gyro = pd.concat([data_gyro_1, data_gyro_2, data_gyro_3, data_gyro_4, data_gyro_5, data_gyro_6, data_gyro_7, data_gyro_8])

    # Save the data
    data.to_csv('datasets/own_data/combined_accelerometer.csv', index=False)
    data_gyro.to_csv('datasets/own_data/combined_gyroscope.csv', index=False)


def analyze_combined_data():
    data = pd.read_csv('datasets/own_data/combined_accelerometer.csv')
    data_gyro = pd.read_csv('datasets/own_data/combined_gyroscope.csv')

    print('data head: ')
    print(data.head())

    print('data gyro head: ')
    print(data_gyro.head())

    print('data info: ')
    print(data.info())

    print('data gyro info: ')
    print(data_gyro.info())

    print('data describe: ')
    print(data.describe())

    print('data gyro describe: ')
    print(data_gyro.describe())

    print('data shape: ')
    print(data.shape)

    print('data gyro shape: ')
    print(data_gyro.shape)

    print('data columns: ')
    print(data.columns)

    print('data gyro columns: ')
    print(data_gyro.columns)

    print('data isna: ')
    print(data.isna().sum())

    print('data gyro isna: ')
    print(data_gyro.isna().sum())

    print('data nunique: ')
    print(data.nunique())

    print('data gyro nunique: ')
    print(data_gyro.nunique())

    print('data value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

    print('data value counts: ')
    print(data['sensor_type'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['sensor_type'].value_counts())

    print('data value counts: ')
    print(data['device_type'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['device_type'].value_counts())

    print('data value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

    print('data value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

    print('data value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

def remove_noise():
    #Remove noise and handle missing values
    data = pd.read_csv('datasets/own_data/combined_accelerometer.csv')
    data_gyro = pd.read_csv('datasets/own_data/combined_gyroscope.csv')

    # Remove noise using crowdsignals_ch3_outliers.py
    # Determine the columns we want to experiment on.
    outlier_columns = ['x', 'y', 'z']
    # Create the outlier classes.
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()
    # Using chauvenet outlier detection, make 1 extra column for each column with outliers
    for col in outlier_columns:
        print(f"Applying Chauvenet outlier criteria for column {col}")
        data = OutlierDistr.chauvenet(data, col, 3)
        data_gyro = OutlierDistr.chauvenet(data_gyro, col, 3)

    # Handle missing values
    data = data.interpolate()
    data_gyro = data_gyro.interpolate()

    # Save the data
    data.to_csv('datasets/own_data/combined_accelerometer_outlier.csv', index=False)
    data_gyro.to_csv('datasets/own_data/combined_gyroscope_outlier.csv', index=False)

    # Plot the data using VisualizeDataset.py, plot_binary_outliers TODO: The x-axis should be the timestamps, the y-axis the values of the columns

    DataViz = VisualizeDataset(__file__)
    print(outlier_columns, 'outlier_columns')
    for col in outlier_columns:

        DataViz.plot_binary_outliers(data, col, col + '_outlier', name_file='combined_accelerometer_outlier')
        DataViz.plot_binary_outliers(data_gyro, col, col + '_outlier', name_file='combined_gyroscope_outlier')





# get_combined_data()
# analyze_combined_data()
remove_noise()
