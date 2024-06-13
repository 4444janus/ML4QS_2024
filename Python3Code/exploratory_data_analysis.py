##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from Python3Code.util.VisualizeDataset import VisualizeDataset
from Python3Code.util import util
from pathlib import Path
import copy
import os
import sys

# label = ['car', 'car', 'car', 'bike', 'walk', 'walk', 'walk', 'tram']
# foldername = ['auto_10hz', 'auto_10hz_v2', 'auto_10hz_v3', 'fietsen_10hz',
#               'lopen_10hz_v1', 'lopen_10hz_v2', 'lopen_10hz_v3', 'tram_10hz_v1']

# Chapter 2: Initial exploration of the dataset.
label = 'bike' # car, bike, walk, tram
foldername = 'fietsen' #['fietsen', 'auto', 'fietsen', 'lopen', 'tram']
version = '10hz' # 10hz, 10hz_v2, 10hz_v3
name_result = f'{label}_{foldername}_{version}'
DATASET_PATH = Path(f'./datasets/own_data/{foldername}_{version}/')
RESULT_PATH = Path('./intermediate_datafiles/')
RESULT_FNAME = f'{label}_result.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

datasets = []
for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    dataset.add_numerical_dataset(f'formatted_{label}_accelerometer.csv', 'timestamps', ['x','y','z'], 'avg', f'acc_{label}_')
    # dataset.add_numerical_dataset('accelerometer_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_watch_')

    # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    dataset.add_numerical_dataset(f'formatted_{label}_gyroscope.csv', 'timestamps', ['x','y','z'], 'avg', f'gyr_{label}_')
    # dataset.add_numerical_dataset('gyroscope_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'gyr_watch_')

    # We add the heart rate (continuous numerical measurements) and aggregate by averaging again
    # dataset.add_numerical_dataset('heart_rate_smartwatch.csv', 'timestamps', ['rate'], 'avg', 'hr_watch_')

    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    # dataset.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')

    # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging
    # dataset.add_numerical_dataset('light_phone.csv', 'timestamps', ['lux'], 'avg', 'light_phone_')

    # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    # dataset.add_numerical_dataset('magnetometer_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_phone_')
    # dataset.add_numerical_dataset('magnetometer_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_watch_')

    # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
    # dataset.add_numerical_dataset('pressure_phone.csv', 'timestamps', ['pressure'], 'avg', 'press_phone_')

    # Get the resulting pandas data table
    dataset = dataset.data_table


    # Plot the data
    DataViz = VisualizeDataset(__file__)
    dataset = dataset.astype(float)
    # print(type(dataset['acc_car_x'][0]))
    # print(dataset.columns)
    # print(dataset.to_string())

    # Boxplot

    DataViz.plot_dataset_boxplot(dataset, [f'acc_{label}_x', f'acc_{label}_y', f'acc_{label}_z', f'gyr_{label}_x', f'gyr_{label}_y', f'gyr_{label}_z'], name_file=name_result+'_'+str(milliseconds_per_instance))

    # Plot all data
    DataViz.plot_dataset(dataset, [f'acc_{label}_x', f'acc_{label}_y', f'acc_{label}_z', f'gyr_{label}_x', f'gyr_{label}_y', f'gyr_{label}_z'],
                                  ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                                  ['line', 'line', 'line', 'line', 'line', 'line', 'line'], name_file=name_result+'_'+str(milliseconds_per_instance))

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')


# Make a table like the one shown in the book, comparing the two datasets produced.
util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through

print('The code has run through successfully!')