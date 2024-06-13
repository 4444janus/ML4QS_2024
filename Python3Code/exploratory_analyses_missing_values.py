import pandas as pd
import numpy as np
from datetime import datetime, timezone
from crowdsignals_ch3_outliers import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection
from util.VisualizeDataset import VisualizeDataset
# Data samenvoegen
# De tijd bij elkaar optellen, dus voor de 2e heb je de huidige tijd per datapunt + eind tijd van 1e

def get_combined_data():
    data_14 = pd.read_csv('datasets/own_data/auto_10hz/formatted_car_accelerometer.csv')
    data_10 = pd.read_csv('datasets/own_data/auto_10hz_v2/formatted_car_accelerometer.csv')
    data_6 = pd.read_csv('datasets/own_data/auto_10hz_v3/formatted_car_accelerometer.csv')
    data_9 = pd.read_csv('datasets/own_data/fietsen_10hz/formatted_bike_accelerometer.csv')
    data_12 = pd.read_csv('datasets/own_data/lopen_10hz_v1/formatted_walk_accelerometer.csv')
    data_3 = pd.read_csv('datasets/own_data/lopen_10hz_v2/formatted_walk_accelerometer.csv')
    data_7 = pd.read_csv('datasets/own_data/lopen_10hz_v3/formatted_walk_accelerometer.csv')
    data_4 = pd.read_csv('datasets/own_data/tram_10hz_v1/formatted_tram_accelerometer.csv')
    data_2 = pd.read_csv('datasets/own_data/tram_10hz_v2/formatted_tram_accelerometer.csv')
    data_8 = pd.read_csv('datasets/own_data/tram_10hz_v3/formatted_tram_accelerometer.csv')
    data_11 = pd.read_csv('datasets/own_data/tram_10hz_v4/formatted_tram_accelerometer.csv')
    data_5 = pd.read_csv('datasets/own_data/fietsen_10hz_v2/formatted_bike_accelerometer.csv')
    data_13 = pd.read_csv('datasets/own_data/fietsen_10hz_v3/formatted_bike_accelerometer.csv')
    data_1 = pd.read_csv('datasets/own_data/fietsen_10hz_v4/formatted_bike_accelerometer.csv')


    data_gyro_14 = pd.read_csv('datasets/own_data/auto_10hz/formatted_car_gyroscope.csv')
    data_gyro_10 = pd.read_csv('datasets/own_data/auto_10hz_v2/formatted_car_gyroscope.csv')
    data_gyro_6 = pd.read_csv('datasets/own_data/auto_10hz_v3/formatted_car_gyroscope.csv')
    data_gyro_9 = pd.read_csv('datasets/own_data/fietsen_10hz/formatted_bike_gyroscope.csv')
    data_gyro_12 = pd.read_csv('datasets/own_data/lopen_10hz_v1/formatted_walk_gyroscope.csv')
    data_gyro_3 = pd.read_csv('datasets/own_data/lopen_10hz_v2/formatted_walk_gyroscope.csv')
    data_gyro_7 = pd.read_csv('datasets/own_data/lopen_10hz_v3/formatted_walk_gyroscope.csv')
    data_gyro_4 = pd.read_csv('datasets/own_data/tram_10hz_v1/formatted_tram_gyroscope.csv')
    data_gyro_2 = pd.read_csv('datasets/own_data/tram_10hz_v2/formatted_tram_gyroscope.csv')
    data_gyro_8 = pd.read_csv('datasets/own_data/tram_10hz_v3/formatted_tram_gyroscope.csv')
    data_gyro_11 = pd.read_csv('datasets/own_data/tram_10hz_v4/formatted_tram_gyroscope.csv')
    data_gyro_5 = pd.read_csv('datasets/own_data/fietsen_10hz_v2/formatted_bike_gyroscope.csv')
    data_gyro_13 = pd.read_csv('datasets/own_data/fietsen_10hz_v3/formatted_bike_gyroscope.csv')
    data_gyro_1 = pd.read_csv('datasets/own_data/fietsen_10hz_v4/formatted_bike_gyroscope.csv')


    # dt = datetime(2024, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
    # timestamp = dt.timestamp()
    # to_subtract = int(timestamp)

    # Make the time stamps for all the data continuous in time. Data 2 should start where data 1 ends, etc.
    # Take into account that timestamps is in unix time, so only add the difference between the last timestamp of the previous data and the first timestamp of the current data
    data_2['timestamps'] = data_2['timestamps'] + data_1['timestamps'].iloc[-1]
    data_3['timestamps'] = data_3['timestamps'] + data_2['timestamps'].iloc[-1]
    data_4['timestamps'] = data_4['timestamps'] + data_3['timestamps'].iloc[-1]
    data_5['timestamps'] = data_5['timestamps'] + data_4['timestamps'].iloc[-1]
    data_6['timestamps'] = data_6['timestamps'] + data_5['timestamps'].iloc[-1]
    data_7['timestamps'] = data_7['timestamps'] + data_6['timestamps'].iloc[-1]
    data_8['timestamps'] = data_8['timestamps'] + data_7['timestamps'].iloc[-1]
    data_9['timestamps'] = data_9['timestamps'] + data_8['timestamps'].iloc[-1]
    data_10['timestamps'] = data_10['timestamps'] + data_9['timestamps'].iloc[-1]
    data_11['timestamps'] = data_11['timestamps'] + data_10['timestamps'].iloc[-1]
    data_12['timestamps'] = data_12['timestamps'] + data_11['timestamps'].iloc[-1]
    data_13['timestamps'] = data_13['timestamps'] + data_12['timestamps'].iloc[-1]
    data_14['timestamps'] = data_14['timestamps'] + data_13['timestamps'].iloc[-1]


    data_gyro_2['timestamps'] = data_gyro_2['timestamps'] + data_gyro_1['timestamps'].iloc[-1]
    data_gyro_3['timestamps'] = data_gyro_3['timestamps'] + data_gyro_2['timestamps'].iloc[-1]
    data_gyro_4['timestamps'] = data_gyro_4['timestamps'] + data_gyro_3['timestamps'].iloc[-1]
    data_gyro_5['timestamps'] = data_gyro_5['timestamps'] + data_gyro_4['timestamps'].iloc[-1]
    data_gyro_6['timestamps'] = data_gyro_6['timestamps'] + data_gyro_5['timestamps'].iloc[-1]
    data_gyro_7['timestamps'] = data_gyro_7['timestamps'] + data_gyro_6['timestamps'].iloc[-1]
    data_gyro_8['timestamps'] = data_gyro_8['timestamps'] + data_gyro_7['timestamps'].iloc[-1]
    data_gyro_9['timestamps'] = data_gyro_9['timestamps'] + data_gyro_8['timestamps'].iloc[-1]
    data_gyro_10['timestamps'] = data_gyro_10['timestamps'] + data_gyro_9['timestamps'].iloc[-1]
    data_gyro_11['timestamps'] = data_gyro_11['timestamps'] + data_gyro_10['timestamps'].iloc[-1]
    data_gyro_12['timestamps'] = data_gyro_12['timestamps'] + data_gyro_11['timestamps'].iloc[-1]
    data_gyro_13['timestamps'] = data_gyro_13['timestamps'] + data_gyro_12['timestamps'].iloc[-1]
    data_gyro_14['timestamps'] = data_gyro_14['timestamps'] + data_gyro_13['timestamps'].iloc[-1]



    # Combine the data
    data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14])
    data_gyro = pd.concat([data_gyro_1, data_gyro_2, data_gyro_3, data_gyro_4, data_gyro_5, data_gyro_6, data_gyro_7, data_gyro_8, data_gyro_9, data_gyro_10, data_gyro_11, data_gyro_12, data_gyro_13, data_gyro_14])

    # Save the data
    data.to_csv('datasets/own_data/combined_accelerometer.csv', index=False)
    data_gyro.to_csv('datasets/own_data/combined_gyroscope.csv', index=False)

    # Convert the data to the format of features below, with the datetime features in datetime format with hours, minutes, seconds and milliseconds
    # features = ['sensor_type','device_type','label','label_start','label_start_datetime','label_end','label_end_datetime']
    # data['label_start_datetime'] = pd.to_datetime(data['timestamps'], unit='ms')

def make_labels_file():
    df = pd.read_csv('datasets/own_data/combined_accelerometer.csv')

    # Set the fixed starting timestamp (June 5, 2024, 12:00 PM)
    start_timestamp = pd.Timestamp('2024-06-05 12:00:00')

    # Create a new column to identify consecutive groups of labels
    df['group_id'] = (df['label'] != df['label'].shift()).cumsum()

    # Group by the group_id column
    grouped_df = df.groupby('group_id')

    # Initialize an empty list to store the interval DataFrames
    interval_dfs = []

    # Initialize start_timestamp with the beginning timestamp of your dataset
    previous_end_time = start_timestamp

    # Iterate over each group
    for i, group in grouped_df:
        # Extract the label, sensor_type, and device_type for the group
        label = group['label'].iloc[0]
        sensor_type = group['sensor_type'].iloc[0]
        device_type = group['device_type'].iloc[0]

        if i == 0:
            # For the first group, use the initialized start_timestamp
            label_start = start_timestamp + pd.to_timedelta(group['timestamps'].iloc[0], unit='s')
        else:
            # For subsequent groups, use the end time of the previous group
            label_start = previous_end_time

        # Calculate the end time of the current group
        label_end = label_start + pd.to_timedelta(group['timestamps'].iloc[-1] - group['timestamps'].iloc[0],
                                                  unit='ns')
        # Update previous_end_time for the next group
        previous_end_time = label_end

        # Calculate the end time of the current group
        label_end = label_start + pd.to_timedelta(group['timestamps'].iloc[-1] - group['timestamps'].iloc[0], unit='ns')

        # Calculate start and end times based on the first timestamp in the group, using milliseconds
        # label_start = start_timestamp + pd.to_timedelta(group['timestamps'].iloc[0], unit='s')
        # label_end = start_timestamp + pd.to_timedelta(group['timestamps'].iloc[-1], unit='s')
        # print(group['timestamps'].iloc[0])
        # print('start', label_start)
        # print(group['timestamps'].iloc[-1])
        # print('end', label_end)

        # Create a new DataFrame for this interval
        interval_df = pd.DataFrame({
            'sensor_type': [sensor_type],
            'device_type': [device_type],
            'label': [label],
            'label_start': [label_start.timestamp() * 1e9],
            'label_start_datetime': [label_start.strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]],
            'label_end': [label_end.timestamp() * 1e9],
            'label_end_datetime': [label_end.strftime('%m/%d/%Y %H:%M:%S.%f')[:-3]]
        })

        # Append this interval DataFrame to the list
        interval_dfs.append(interval_df)

        # Update the start timestamp for the next group based on the end time of this group
        start_timestamp = label_end

    # Concatenate all interval DataFrames
    result_df = pd.concat(interval_dfs, ignore_index=True)

    pd.set_option('display.max_columns', None)
    print(result_df)

    # Save the data
    result_df.to_csv('datasets/own_data/labels.csv', index=False)


def make_unix():
    # df = pd.read_csv('datasets/own_data/combined_accelerometer.csv')
    df_gyro = pd.read_csv('datasets/own_data/combined_gyroscope.csv')
    # Change the timestamps to unix time
    # df['timestamps'] = pd.to_datetime(df['timestamps'], unit='s')
    # start_timestamp = pd.Timestamp('2024-06-05 12:00:00')
    # df['timestamps'] = start_timestamp + pd.to_timedelta(df['timestamps'], unit='s')
    # # now change it to unix
    # Initial start timestamp
    start_timestamp = pd.Timestamp('2024-06-05 12:00:00')

    # Convert the 'timestamps' column to timedeltas and add to the start timestamp
    df_gyro['absolute_timestamp'] = start_timestamp + pd.to_timedelta(df_gyro['timestamps'], unit='s')
    df_gyro['timestamps'] = df_gyro['absolute_timestamp'].apply(lambda x: x.timestamp() * 1e9)
    df_gyro.drop(['absolute_timestamp'], axis=1, inplace=True)

    print(df_gyro['timestamps'].head())
    # df['timestamps'] = df['timestamps'].astype('int64') // 10**9
    # print(df['timestamps'].head())
    # df_gyro['timestamps'] = pd.to_datetime(df_gyro['timestamps'], unit='s')
    # df_gyro['timestamps'] = df_gyro['timestamps'].astype('int64') // 10**9
    # df.to_csv('datasets/own_data/combined_accelerometer.csv', index=False)
    df_gyro.to_csv('datasets/own_data/combined_gyroscope.csv', index=False)


def analyze_combined_data():
    data = pd.read_csv('datasets/own_data/combined_accelerometer.csv')
    data_gyro = pd.read_csv('datasets/own_data/combined_gyroscope.csv')

    print('data accelerometer head: ')
    print(data.head())

    print('data gyro head: ')
    print(data_gyro.head())

    print('data accelerometer info: ')
    print(data.info())

    print('data gyro info: ')
    print(data_gyro.info())

    print('data accelerometer describe: ')
    print(data.describe())

    print('data gyro describe: ')
    print(data_gyro.describe())

    print('data accelerometer shape: ')
    print(data.shape)

    print('data gyro shape: ')
    print(data_gyro.shape)

    print('data accelerometer columns: ')
    print(data.columns)

    print('data gyro columns: ')
    print(data_gyro.columns)

    print('data isna: ')
    print(data.isna().sum())

    print('data gyro isna: ')
    print(data_gyro.isna().sum())

    print('data accelerometer nunique: ')
    print(data.nunique())

    print('data gyro nunique: ')
    print(data_gyro.nunique())

    print('data accelerometer value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

    print('data accelerometer value counts: ')
    print(data['sensor_type'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['sensor_type'].value_counts())

    print('data accelerometer value counts: ')
    print(data['device_type'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['device_type'].value_counts())

    print('data accelerometer value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

    print('data accelerometer value counts: ')
    print(data['label'].value_counts())

    print('data gyro value counts: ')
    print(data_gyro['label'].value_counts())

    print('data accelerometer value counts: ')
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
    # Using chauvenet outlier detection, make 1 extra column for each column with outliers
    for col in outlier_columns:
        print(f"Applying Chauvenet outlier criteria for column {col}")
        data = OutlierDistr.chauvenet(data, col, 3)
        data_gyro = OutlierDistr.chauvenet(data_gyro, col, 3)

    # Handle missing values
    # data_chauv = data_chauv.interpolate()
    # data_gyro_chauv = data_gyro_chauv.interpolate()

    # Save the data
    # data_chauv.to_csv('datasets/own_data/combined_accelerometer_outlier_chauv.csv', index=False)
    # data_gyro_chauv.to_csv('datasets/own_data/combined_gyroscope_outlier_chauv.csv', index=False)
    #
    # data_dist = pd.read_csv('datasets/own_data/combined_accelerometer.csv')
    # data_gyro_dist = pd.read_csv('datasets/own_data/combined_gyroscope.csv')

    # Now use simple based outlier detection
    OutlierDist = DistanceBasedOutlierDetection()
    print(data.info())
    # Using simple based outlier detection
    for col in outlier_columns:
        print(f"Applying simple based outlier criteria for column {col}")
        data = OutlierDist.simple_distance_based(data, [col], 'euclidean', 0.1, 0.99)
        data_gyro = OutlierDist.simple_distance_based(data_gyro, [col], 'euclidean', 0.1, 0.99)

    # check if contains missing values
    # print('data isna: ')
    # print(data.isna().sum())

    # Handle missing values
    # data_dist = data_dist.interpolate()
    # data_gyro_dist = data_gyro_dist.interpolate()

    # Save the data
    data.to_csv('datasets/own_data/combined_accelerometer_outlier.csv', index=False)
    data_gyro.to_csv('datasets/own_data/combined_gyroscope_outlier.csv', index=False)

    # Plot the data using VisualizeDataset.py, plot_binary_outliers TODO: The x-axis should be the timestamps, the y-axis the values of the columns

    DataViz = VisualizeDataset(__file__)
    print(outlier_columns, 'outlier_columns')
    for col in outlier_columns:

        DataViz.plot_binary_outliers('Simple-dist accelerometer', data, col, 'simple_dist_outlier', name_file='combined_accelerometer_outlier')
        DataViz.plot_binary_outliers('Simple-dist gyroscope', data_gyro, col, 'simple_dist_outlier', name_file='combined_gyroscope_outlier')
        DataViz.plot_binary_outliers('Chauvenet accelerometer', data, col, col + '_outlier', name_file='combined_accelerometer_outlier')
        DataViz.plot_binary_outliers('Chauvenet gyroscope', data_gyro, col, col + '_outlier', name_file='combined_gyroscope_outlier')


# get_combined_data()
# analyze_combined_data()
# remove_noise()
# make_unix()
make_labels_file()