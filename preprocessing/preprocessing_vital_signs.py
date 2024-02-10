import os
import argparse
import pandas as pd
import numpy as np
from dateutil.rrule import rrule, MINUTELY
import warnings
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from load_raw_vital_signs import *
from preprocess_outputs import preprocess_crt_avpu

CLINICAL_DATA_PATH = '/DATA/Clean_data/IMPALA_Clinical_data_raw.csv'
MISSING = np.nan


def get_age(path: str) -> dict:
    """
    Create a dictionary of all patients with corresponding age in months.
    :param path: string containing path to the clincal data.
    """

    df = pd.read_csv(path, low_memory=False, usecols=['record_id', 'recru_age_months'])

    return {record_id : values['recru_age_months'].iloc[0] for \
            record_id, values in df.groupby('record_id')}


def split_data_into_window(df: pd.DataFrame, time_freq: int = 15) -> (list, list):
    """
    Split the data into windows.
    :param df: Pandas DataFrame containing the data indexed on timestamps.
    :param time_unit: time unit of the data window, e.g. s (seconds), m (minutes).
    :param time_freq: number of time units in the data window.
    """

    windows = []
    datetimes = []
    num_features = df.shape[1]

    for start in rrule(freq=MINUTELY, interval=time_freq,
                       dtstart=df['datetime'].iloc[0], until=df['datetime'].iloc[-1]):

        # Select window
        end = start + pd.Timedelta(time_freq, unit='m')
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


    datetimes = np.array(datetimes)

    return windows, datetimes


def aggregate_windows(windows: list) -> np.ndarray:
    """
    Aggregate vital signs.

    Suppress 'Mean of empty slice' or 'All-NaN slice encountered' warnings as
    these are dealt with manually.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Calculate mean, min, max and std of ECGHR, ECGRR, SPO2HR, SPO2
        mean_ = np.array([np.nanmean(a[:, :4], axis=0) for a in windows])
        min_ = np.array([np.nanmin(a[:, :4], axis=0) for a in windows])
        max_ = np.array([np.nanmax(a[:, :4], axis=0) for a in windows])
        std_ = np.array([np.nanstd(a[:, :4], axis=0) for a in windows])

    # Replace NaN values with missing value token
    mean_ = np.where(np.isnan(mean_), MISSING, mean_)
    min_ = np.where(np.isnan(min_), MISSING, min_)
    max_ = np.where(np.isnan(max_), MISSING, max_)
    std_ = np.where(np.isnan(std_), MISSING, std_)

    # Choose last valid entry of NIBP_lower, NIBP_upper, NIBP_mean, Hour
    others = []
    for w in windows:
        current = []
        for i in range(4):
            valid_entries = w[:, 4+i][np.where(~np.isnan(w[:, 4+i]))]
            current.append(valid_entries[-1] if valid_entries.size > 0 else MISSING)
        others.append(current)
    others = np.array(others)

    return np.concatenate([mean_, min_, max_, std_, others], axis=1)


def standardize(data: np.ndarray) -> np.ndarray:
    """ Standardize the data per feature. """
    return StandardScaler().fit_transform(data)


def normalize(data: np.ndarray) -> np.ndarray:
    """ Normalize the data between 0 and 1. """
    return MinMaxScaler().fit_transform(data)


def sliding_window(args, data: np.ndarray, output: pd.DataFrame, datetimes: list) -> (np.ndarray, np.ndarray):
    """
    Apply a sliding window over the given data. If a window is too long, the first
    entries are discarded. If a window is too short it is discarded completely.
    Add overlap parameter to control how much the windows should overlap.

    :param data: NumPy Array containing the aggregated vital sign windows.
    :param datetimes: List containing the starting times for each window.
    :param output: Pandas Dataframe containing CRT and AVPU values per time.
    :param args.window_hours: Desired length of the sliding windows in hours.
    :param args.overlap: Ratio of overlap between sliding windows.
    :param args.output_deviation: Maximum difference between end of the window and output time. 
    """

    X, y = [], []
    datetimes = pd.Series(datetimes, name='time')

    if args.overlap < 1:
        overlap_hours = round(args.window_length * 4 * (1 - args.overlap))
    else:
        overlap_hours = 1

    for i in range(len(datetimes)-1, -1, -overlap_hours):
        
        # Create window
        end = datetimes[i]
        start = end - pd.Timedelta(args.window_length, unit='h')
        idx = datetimes.between(start, end)
        window = data[idx]

        # Window size contraints
        optimal_window_size = int(args.window_length * 4)
        if window.shape[0] < optimal_window_size:
            continue
        elif window.shape[0] > optimal_window_size:
            window = window[-optimal_window_size:]

        # Choose corresponding output (nearest to endtime and within X hours)
        time_diff = [abs(end - t) for t in output.index]
        diff_within_range = [i for i, t in enumerate(time_diff) if t < pd.Timedelta(args.deviation, unit='h')]
        nearest_i = np.argmin(time_diff) if np.argmin(time_diff) in diff_within_range else None

        if nearest_i is not None:
            X.append(window)
            y.append(output.iloc[nearest_i].values)
    
    X = np.array(X)
    y = np.array(y)

    return X, y


def preprocessing(args):
    """
    Preprocess the vital sign data and return a dictionary containing the vital
    sign windows and corresponding output (value) per patient(key).
    """

    withdrew_consent = ['B-N-0001', 'B-N-0002', 'B-N-0006', 'B-N-0015', 'B-N-0018',
                        'B-N-0020', 'B-N-0026', 'B-N-0056', 'B-N-0094', 'B-S-0010',
                        'B-S-0048', 'B-S-0178', 'B-S-0307']
    
    absconded = ['B-N-0067', 'B-N-0075', 'B-S-0086', 'B-S-0153', 'B-S-0155',
                 'B-S-0173', 'B-S-0186', 'B-S-0190', 'B-S-0285', 'B-S-0291',
                 'B-S-0297', 'B-S-0299', 'Z-H-0103']

    file_path = os.getcwd() + args.data_dir
    if args.test:
        data = load_patient_dict(file_path)
    else:
        data = read_raw_vital_signs(file_path)

    output_dict = preprocess_crt_avpu(os.getcwd() + CLINICAL_DATA_PATH)
    age_dict = get_age(os.getcwd() + CLINICAL_DATA_PATH)

    patient_data = {}

    for patient_id, df in data.items():

        if patient_id in withdrew_consent:
            continue

        if patient_id not in age_dict or patient_id not in output_dict:
            print(f'Patient {patient_id} not found in age or output data')
            continue

        if df.empty:
            print(f'Patient {patient_id} has not vital sign data')
            continue

        # Split data into 15 minute windows
        df = df.drop(['patient_id', 'location'], axis=1)
        df.sort_values('datetime', inplace=True)
        windows, datetimes = split_data_into_window(df)

        # Aggregate windows accross time
        agg_data = aggregate_windows(windows)
        age_data = (np.ones((agg_data.shape[0], 1)) * age_dict[patient_id])
        new_data = np.concatenate((agg_data, age_data), axis=1)
        
        if args.standardize:
            new_data = standardize(new_data)

        if args.normalize:
            new_data = normalize(new_data)

        # Sliding window
        X, y = sliding_window(args, new_data, output_dict[patient_id], datetimes)
        
        patient_data[patient_id] = {'X' : X, 'y' : y}

    
    if args.verbose:
        n_patients = len(patient_data)
        n_samples = sum([a["y"].shape[0] for a in patient_data.values()])
        sample_length = X.shape[1]
        n_features = X.shape[-1]

        missing = np.zeros(n_features)
        for d in patient_data.values():
            missing += np.sum(np.isnan(d['X']), axis=(0, 1))


        print('=== Data statistics ===')
        print(f' No patients     : {n_patients}')
        print(f' No data samples : {n_samples}')
        print(f' Sample length   : {sample_length}')
        print(f' No features     : {n_features}')
        
        features = ["ECGHR_mean", "ECGRR_mean", "SPO2HR_mean", "SPO2_mean",
                    "ECGHR_min", "ECGRR_min", "SPO2HR_min", "SPO2_min",
                    "ECGHR_max", "ECGRR_max", "SPO2HR_max", "SPO2_max",
                    "ECGHR_std", "ECGRR_std", "SPO2HR_std", "SPO2_std",
                    "NIBP_lower", "NIBP_upper", "NIBP_mean", "Hour", "Age"]

        print('\n=== Missing data per feature (%) ===')
        max_len = max(len(f) for f in features)
        for f, p in zip(features, (missing / (n_samples * sample_length) * 100).round(2)):
            print(f" {f}{' '*(max_len-len(f))} : {p}")

    return patient_data
        

def main(args):

    data = preprocessing(args)

    if args.results_dir:
        filename = f'VitalSignDataset_w{args.window_length}_d{args.deviation}'
        if args.overlap: filename += f'_o{args.overlap}'
        if args.standardize: filename += f'_standardize'
        if args.normalize: filename += f'_normalize'

        filename += '.pkl'

        with open(os.getcwd() + args.results_dir + filename, 'wb') as f:
            pickle.dump(data, f)
    
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true',
                        help='show additional informaton')
    parser.add_argument('--test', action='store_true',
                        help='use test dataset of 30 patients (instead of full dataset)')
    parser.add_argument('--data_dir', action='store', type=str, required=True,
                        help="path to data directory")
    parser.add_argument('--results_dir', action='store', type=str,
                        help='path to directory to save the preprocessed data to')
    
    # Data processing
    parser.add_argument('--standardize', action='store_true',
                        help='standardize the data')
    parser.add_argument('--normalize', action='store_true',
                        help='normalize the data')

    # Sliding window arguments
    parser.add_argument('-w', '--window_length', type=int, required=True,
                        action='store', help='size of sliding window in hours')
    parser.add_argument('-o', '--overlap', type=float, action='store',
                        help='ratio which sliding windows can overlap')
    parser.add_argument('-d', '--deviation', type=int, action='store',
                        required=True, help='maximum output deviation (hours)')
    
    args = parser.parse_args()

    assert args.overlap >= 0 and args.overlap <= 1, 'overlap must be between 0 and 1'
    assert not(args.standardize and args.normalize), 'can only select standardize or normalize'

    _ = main(args)