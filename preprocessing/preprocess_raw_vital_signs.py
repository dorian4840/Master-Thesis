import os
import argparse
import pandas as pd
import numpy as np
from dateutil.rrule import rrule, SECONDLY, MINUTELY, HOURLY
import warnings
from itertools import product

from heart_rate_variability import apply_hrv
from load_raw_vital_signs import *
from preprocess_outputs import preprocess_crt_avpu
from datasets import create_dataloaders

RAW_VITAL_DATA_PATH = "/DATA/Raw Data/filtered_df_removed_nan_files.parquet"
CLINICAL_DATA_PATH = "/DATA/Clean Data/IMPALA_Clinical_Data_202308211019_Raw.csv"
PROCESSED_RAW_VITAL_SIGN_DATA_PATH = "/DATA/Raw Data/raw_patient_dict_p30"


def get_age_in_months(path):
    """
    Create a dictionary of all patients with corresponding age in months.

    :param path: string containing path to the clincal data.
    """

    df = pd.read_csv(path, low_memory=False, usecols=['record_id', 'recru_age_months'])

    return {record_id : values['recru_age_months'].iloc[0] for \
            record_id, values in df.groupby('record_id')}


def split_data_into_window(df, time_unit='m', time_freq=15):
    """
    Split the data into windows.
    :param df: Pandas DataFrame containing the data indexed on timestamps.
    :param time_unit: time unit of the data window, e.g. s (seconds), m (minutes).
    :param time_freq: number of time units in the data window.
    """

    df = df.copy()

    rrule_time = {'h' : HOURLY, 'm' : MINUTELY, 's' : SECONDLY}
    windows = []
    datetimes = []
    num_features = df.shape[1]

    for start in rrule(freq=rrule_time[time_unit], interval=time_freq,
                       dtstart=df['datetime'].iloc[0], until=df['datetime'].iloc[-1]):
        
        # Select window
        end = start + pd.Timedelta(time_freq, unit=time_unit)
        idx = df['datetime'].between(start, end)
        window = df[idx]

        if window.size > 0:
            # From datetime only keep hours
            window.loc[:, 'datetime'] = window.loc[:, 'datetime'].dt.hour

            # Save windows and timepoints seperately
            windows.append(window.values)
            datetimes.append(df[idx]['datetime'].iloc[0])

        else: # If no data in time window, still add empty window
            window = np.empty((1, num_features))
            window.fill(np.float64('nan'))
            window[0, -1] = end.hour

            # Save windows and timepoints seperately
            windows.append(window)
            datetimes.append(start)

    return windows, np.array(datetimes)


def aggregate_windows(windows):
    """
    Aggregate vital signs.

    Suppress 'Mean of empty slice' or 'All-NaN slice encountered' warnings as
    these are dealt with manually.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Calculate mean, min, max and std of the first four features
        mean_ = np.array([np.nanmean(a[:, :4], axis=0) for a in windows])
        min_ = np.array([np.nanmin(a[:, :4], axis=0) for a in windows])
        max_ = np.array([np.nanmax(a[:, :4], axis=0) for a in windows])
        std_ = np.array([np.nanstd(a[:, :4], axis=0) for a in windows])

    # Replace NaN values with -999
    mean_ = np.where(np.isnan(mean_), -999, mean_)
    min_ = np.where(np.isnan(min_), -999, min_)
    max_ = np.where(np.isnan(max_), -999, max_)
    std_ = np.where(np.isnan(std_), -999, std_)

    # Choose latest valid entries of last four features
    other_features = []
    for i, w in enumerate(windows):

        current_window = []
        bool_labels = np.where(np.isnan(w[:, 4:]), -999, w[:, 4:]) >= 0

        for i in range(4):
            valid_entries = w[:, i+4][bool_labels[:, i]]
            current_window.append(valid_entries[-1] if valid_entries.size > 0 else -999)

        other_features.append(current_window)
    
    other_features = np.array(other_features)

    return np.concatenate([mean_, min_, max_, std_, other_features], axis=1)


def sliding_window_backward(data, outputs, datetimes, sample_window_hours):
    """
    Apply a sliding window over the given data. If a window is too long, the first
    entries are discarded. If a window is too short it is discarded completely.

    :param data: NumPy Array containing the aggregated vital sign windows.
    :param datetimes: List containing the starting times for each window.
    :param outputs: Pandas Dataframe containing CRT and AVPU values per time.
    """

    X, y = [], []

    datetimes = pd.Series(datetimes, name='time')

    for end in datetimes[::-1]:

        start = end - pd.Timedelta(sample_window_hours, unit='h')
        idx = datetimes.between(start, end)
        window = data[idx]


        if window.shape[0] < sample_window_hours * 4:
            continue

        elif window.shape[0] > sample_window_hours * 4: # Remove first entry
            window = window[-sample_window_hours*4:]
        

        # Choose output that is nearest to the end time of the window
        nearest_output_idx = np.argmin([abs(end - t) for t in outputs.index])

        X.append(window.T)
        y.append(outputs.iloc[nearest_output_idx].values)

    return X, y


def main(args):
    """
    Preprocess the raw vital sign data.
    """

    data = load_patient_dict(os.getcwd() + PROCESSED_RAW_VITAL_SIGN_DATA_PATH)
    outputs = preprocess_crt_avpu(os.getcwd() + CLINICAL_DATA_PATH)
    age_dict = get_age_in_months(os.getcwd() + CLINICAL_DATA_PATH)

    X, y = [], []

    for patient_id, df in data.items():

        ### 1. Split data into windows ###
        df = data['Z-H-0120'].drop(['patient_id', 'location'], axis=1)
        df = df.sort_values('datetime')
        windows, datetimes = split_data_into_window(df, time_unit='m', time_freq=15)

        ### 2. Transform windows
        agg_data = aggregate_windows(windows) # Aggragate over windows
        hrv_data = apply_hrv(windows, return_features=args.hrv) # Calculate HRV
        age_data = (np.ones((agg_data.shape[0], 1)) * age_dict[patient_id]) # Add age
        new_data = np.concatenate((agg_data, hrv_data, age_data), axis=1)

        ### 3. Apply sliding window
        temp_X, temp_y = sliding_window_backward(new_data, outputs[patient_id],
                                                 datetimes, args.window_size)
        X.append(temp_X)
        y.append(temp_y)


    X = [l for s in X for l in s]
    y = [l for s in y for l in s]

    X = np.array(X) # Shape: samples, dimensions, time
    y = np.array(y) # Shape: samples, dimensions

    if args.verbosity:
        features = ["ECGHR_mean,", "ECGRR_mean,", "SPO2HR_mean,", "SPO2_mean,\n",
                    "ECGHR_min,", "ECGRR_min,", "SPO2HR_min,", "SPO2_min,\n",
                    "ECGHR_max,", "ECGRR_max,", "SPO2HR_max,", "SPO2_max,\n",
                    "ECGHR_std,", "ECGRR_std,", "SPO2HR_std,", "SPO2_std,\n"]
        print("=== Input features ===\n", *features, *args.hrv, "Age\n")
        print("=== Output features ===\n CRT, AVPU\n")
        print(f"Input shape: {X.shape}\noutput shape: {y.shape}")

    ### 4. Load data into dataloaders ###
    train_dataloader, val_dataloader, test_dataloader = \
        create_dataloaders(X, y, batch_size=args.batch_size, seed=args.seed)
    
    return train_dataloader, val_dataloader, test_dataloader


def list_of_strings(arg):
    return [s.strip() for s in arg.split(',')]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", action="store_true",
                        help="if set, print information regarding the data")
    parser.add_argument("-w", "--window_size", type=int, required=True, action="store",
                        help="set the size of the sliding windows (required)")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, action="store",
                        help="set the batch size of the dataloaders (required)")
    parser.add_argument("-s", "--seed", type=int, required=True, action="store",
                        help="set the seed of the dataloaders (required)")
    parser.add_argument("--hrv", type=list_of_strings, required=True, action="store",
                        help="list the hrv features to be calculated (default: lfnu)")
    
    args = parser.parse_args()

    train_dataloader, val_dataloader, test_dataloader = main(args)