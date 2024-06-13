import pandas as pd
from datetime import datetime, timezone


def change_dataset(location='datasets/own_data/auto_10hz_v3', dataset='Accelerometer.csv', label='auto', sensor_type='accelerometer'):
    """change the raw data to look like the data from table 2.3 in page 17 of the book"""
    #time at yeesturday

    dt = datetime(2024, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
    timestamp = dt.timestamp()
    print(int(timestamp))
    # Read the data
    data = pd.read_csv(location + '/' + dataset)

    # Rename the columns
    data.columns = ['timestamps', 'x', 'y', 'z']
    print(data['timestamps'])
    # print(data.to_string())
    #add unix time to time
    # data['timestamps'] = (data['timestamps'] + int(timestamp))
    print(data['timestamps'])
    # data['datetime'] = pd.to_datetime(data['timestamps'], unit='s')
    # data.drop(['timestamps'], axis=1, inplace=True)
    # data['timestamps'] = data['datetime'].astype('int64') // 10**9
    # data.drop(['datetime'], axis=1, inplace=True)
    # print(data.info())

    #add the labels
    data['label'] = label
    data['sensor_type'] = sensor_type
    data['device_type'] = 'smartphone'

    #make float
    data['x'] = data['x'].astype(float)
    data['y'] = data['y'].astype(float)
    data['z'] = data['z'].astype(float)

    # Save the data
    data.to_csv(location + '/' + f'formatted_{label}_{sensor_type}.csv', index=False)
    # print(data.to_string())
    print(data.head())
    return data

def make_labels_file(dataset='combined_accelerometer.csv', location='datasets/own_data'):
    #sensor_type,device_type,label,label_start,label_start_datetime,label_end,label_end_datetime
    print('make labels file')
    data = pd.read_csv(location + '/' + dataset)
    print(data.head())
    #add label_start and label_end of labels

    # data['label_start'] = data['timestamps']
# make_labels_file()



# change_dataset()
# auto1
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='car', location='datasets/own_data/auto_10hz')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='car', location='datasets/own_data/auto_10hz')
#auto2
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='car', location='datasets/own_data/auto_10hz_v2')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='car', location='datasets/own_data/auto_10hz_v2')
#auto3
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='car', location='datasets/own_data/auto_10hz_v3')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='car', location='datasets/own_data/auto_10hz_v3')
#fietsen
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='bike', location='datasets/own_data/fietsen_10hz')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='bike', location='datasets/own_data/fietsen_10hz')
#fietsen2
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='bike', location='datasets/own_data/fietsen_10hz_v2')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='bike', location='datasets/own_data/fietsen_10hz_v2')
#fietsen3
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='bike', location='datasets/own_data/fietsen_10hz_v3')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='bike', location='datasets/own_data/fietsen_10hz_v3')
#fietsen4
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='bike', location='datasets/own_data/fietsen_10hz_v4')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='bike', location='datasets/own_data/fietsen_10hz_v4')
#lopen1
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='walk', location='datasets/own_data/lopen_10hz_v1')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='walk', location='datasets/own_data/lopen_10hz_v1')
#lopen2
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='walk', location='datasets/own_data/lopen_10hz_v2')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='walk', location='datasets/own_data/lopen_10hz_v2')
#lopen3
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='walk', location='datasets/own_data/lopen_10hz_v3')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='walk', location='datasets/own_data/lopen_10hz_v3')
#tram
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='tram', location='datasets/own_data/tram_10hz_v1')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='tram', location='datasets/own_data/tram_10hz_v1')
#tram2
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='tram', location='datasets/own_data/tram_10hz_v2')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='tram', location='datasets/own_data/tram_10hz_v2')
#tram3
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='tram', location='datasets/own_data/tram_10hz_v3')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='tram', location='datasets/own_data/tram_10hz_v3')
#tram4
change_dataset(dataset='Accelerometer.csv', sensor_type='accelerometer', label='tram', location='datasets/own_data/tram_10hz_v4')
change_dataset(dataset='Gyroscope.csv', sensor_type='gyroscope', label='tram', location='datasets/own_data/tram_10hz_v4')
