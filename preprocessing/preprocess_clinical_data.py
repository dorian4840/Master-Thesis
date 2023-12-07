import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import torch

from preprocess_outputs import preprocess_crt_avpu
from datasets import create_dataloaders, IMPALA_Dataset

CLINICAL_DATA_PATH = '/DATA/Clean Data/IMPALA_Clinical_Data_202308211019_Raw.csv'
CLINICAL_DTYPES_PATH = '/DATA/IMPALAclinicalstudy_Dictionary_2023-09-25.csv'


### Load data
def read_clinical_df(path):
    """ Load clinical data into a Pandas DataFrame. """
    return pd.read_csv(path, low_memory=False)


def read_clinical_dtype_dict(path):
    """ Load clinical dict containing all variable names and their properties. """

    df = pd.read_csv(path)
    df = df[df['Variable'].str.startswith(tuple(['recru', 'dly', 'record_id']))]

    dtype_dict = defaultdict(list)
    for _, row in df.iterrows():
        dtype_dict[row['Field Type']].append(row['Variable'])
    
    del df

    return dtype_dict


### Clean data
def intersection(l1, l2):
    """ Create intersection of two lists. """
    return [v1 for v1 in l1 if v1 in l2]


def split_per_dtype(c_df, d_dict):
    """
    Split the clinical dataset per datatype.
    """

    c_df = c_df.copy()  # Might remove later
    d_dict = d_dict.copy() # Might remove later

    # Remove rows that have no daily time (as those are duplicates of the row above)
    c_df = c_df[-pd.isna(c_df['dly_time'])]
    c_df.reset_index(inplace=True)

    # Replace missing values with -1
    c_df.replace(to_replace=99, value=-1, inplace=True)
    c_df.fillna(-1, inplace=True)

    # Only select columns that appear in clinical data
    dtype_columns = {k : intersection(v, c_df.columns.to_list()) for k, v in d_dict.items()}
    dtype_df = {k : c_df[v] for k, v in dtype_columns.items()}


    # 'recru_cr_time_result', 'dly_crt', 'dis_cr_time']] < 0] = np.nan
    # df[df[['recru_avpu_score', 'dly_avpu

    return dtype_df


def remove_columns(dtype_df, threshold, args):
    """ Remove columns if the ratio of missing values exceeds the threshold. """

    if args.verbosity:
        print(f"Remove columns if too many values are missing:")

    for type_, df in dtype_df.items():
        drop = []

        for col in df:
            if -1 in df[col].value_counts(normalize=True).to_dict().keys() and \
                    df[col].value_counts(normalize=True).to_dict()[-1] > threshold:

                drop.append(col)
        
        if args.verbosity:
            print(f'{type_}: removed {len(drop)} columns')

        dtype_df[type_] = dtype_df[type_].drop(drop, axis=1)
    
    return dtype_df


def split_text_data(dtype_df):
    """
    Split the text data into numerical, record_id, dates and text
    """

    dtype_df = dtype_df.copy()

    # Split numerical columns
    if 'num' not in dtype_df.keys():
        num_df = dtype_df['text'].select_dtypes(include=['float64', 'int64'])
        valid_num_columns = num_df.columns.to_list()
        dtype_df['text'] = dtype_df['text'].drop(valid_num_columns, axis=1)
        dtype_df['num'] = num_df

    # Split record_id
    if 'record_id' in dtype_df['text'].columns:
        record_id = dtype_df['text']['record_id']
        dtype_df['text'] = dtype_df['text'].drop('record_id', axis=1)
        dtype_df['record_id'] = record_id

    # Split date columns (columns handpicked)
    if 'dates' not in dtype_df.keys():
        date_df = dtype_df['text'].filter(
            ["recru_interview_date_", "recru_hospital_admission_date",
            "recru_hdu_admission_date", "recru_dob", "recru_bloodculture_time",
            "recru_lactate_sample_time", "recru_storage_spec_time",
            "recru_nasal_swab_time", "recru_urine_time", "dly_time",
            "dly_time_new_cie1a", "dly_time_new_cie2", "dly_time_new_cie3",
            "dly_time_new_cie4", "dly_time_new_cie5", "dly_time_new_cie6"])
        
        dtype_df['dates'] = date_df
        dtype_df['text'] = dtype_df['text'].drop(date_df.columns, axis=1)

    return dtype_df


def one_hot_encoding(df):
    """
    Convert categorical columns to multiple binary columns and add them to the
    end of the DataFrame. Remove binary columns that are created for
    missing/NaN values.
    """

    df = df.copy()

    categorical_columns = df.columns.to_list()
    drop = []

    for col in categorical_columns:

        if -1 in df[col].value_counts().to_dict().keys():
            DROP_FIRST = True

        else:
            DROP_FIRST = False

        df = df.join(pd.get_dummies(df[col],
                                    dummy_na=False, 
                                    prefix=col,
                                    sparse=True,
                                    drop_first=DROP_FIRST,
                                    dtype=float))

        # Drop original categorical column
        drop.append(col)

    df = df.drop(drop, axis=1)

    return df


### PCA
def perform_PCA(data, visualize=False):
    """
    Normalize the data and perform PCA to reduce dimensionality.
    If visualize is True, plot the explained variance per PC.
    """

    # Perform Principal Component Analysis
    pca = PCA(n_components='mle')
    pca.fit(data)
    new_data = pca.transform(data)

    if visualize:
        plt.plot(range(pca.n_components_), pca.explained_variance_ratio_)
        plt.title('Explained variance ratio per principal component')
        plt.xlabel('Number of components')
        plt.ylabel('Ratio of explained variance')
        plt.legend(['Explained Variance', "Number of PCs found using Minka's MLE"])
        plt.tight_layout()
        plt.show()

    return new_data


### Sliding windpw
def sliding_window_backward(data, outputs, datetimes, admission_dates, record_ids, sample_window_hours):
    """
    Apply a sliding window over the given data. If a window is too long, the first
    entries are discarded. If a window is too short it is discarded completely.

    :param data: NumPy Array containing the clinical data.
    :param outputs: Pandas Dataframe containing CRT and AVPU values per time.
    :param datetimes: Pandas DataFrame containing the time at each data point.
    :param record_ids: Pandas DataFrame containing the record IDs per patient.
    """
    
    X, y = [], []

    for record_id, df in record_ids.groupby(by='record_id', observed=True):

        idx = df.index
        del df

        curr_data = data[idx, :]
        curr_output = outputs[record_id]
        curr_datetime = datetimes.iloc[idx]

        curr_data = np.concatenate((curr_data[0:1, :], curr_data))
        curr_datetime = pd.concat([pd.Series(admission_dates[record_id]), curr_datetime],
                                  ignore_index=True)
        curr_datetime = pd.to_datetime(curr_datetime, format='mixed')

        for end in curr_datetime[::-1]:

            start = end - pd.Timedelta(sample_window_hours, unit='h')
            idx = curr_datetime.between(start, end)
            window = curr_data[idx]

            if window.shape[0] < (sample_window_hours / 4) + 1:
                continue

            elif window.shape[0] > (sample_window_hours / 4) + 1: # Remove first entry
                window = window[-int((sample_window_hours / 4) + 1):]
            
            # Calculate to which output the end of the window lies closest
            nearest_output_idx = np.argmin( [abs(end - t) for t in curr_output.index] )

            X.append(window.T)
            y.append(curr_output.iloc[nearest_output_idx].values)

    X = np.array(X) # Shape: samples, dimensions, time
    y = np.array(y) # Shape: samples, dimensions

    return X, y


def sliding_window_forward(data, outputs, datetimes, admission_dates, record_ids, sample_window_hours):
    """
    Apply a sliding window over the given data. If a window is not of the correct
    size it is discarded.

    :param data:     NumPy Array containing the clinical data.
    :param outputs:  Pandas Dataframe containing CRT and AVPU values per time.
    :param datetimes: Pandas DataFrame containing the time at each data point.
    :param record_ids: Pandas DataFrame containing the record IDs per patient.
    """
    
    X, y = [], []

    for record_id, df in record_ids.groupby(by='record_id', observed=True):

        idx = df.index
        del df
        curr_data = data[idx, :]
        curr_output = outputs[record_id]
        curr_datetime = datetimes.iloc[idx]

        curr_data = np.concatenate((curr_data[0:1, :], curr_data))
        curr_datetime = pd.concat([pd.Series(admission_dates[record_id]), curr_datetime], ignore_index=True)
        curr_datetime = pd.to_datetime(curr_datetime, format='mixed')

        for start in curr_datetime:

            end = start + pd.Timedelta(sample_window_hours, unit='h')
            idx = curr_datetime.between(start, end)
            window = curr_data[idx]

            # Only continue if window is the appropriate size (5016 windows)
            if window.shape[0] == (sample_window_hours / 4) + 1:

                # Calculate to which output the end of the window lies closest
                nearest_output_idx = np.argmin( [abs(end - t) for t in curr_output.index] )

                X.append(window.T)
                y.append(curr_output.iloc[nearest_output_idx].values)


    X = np.array(X) # Shape: samples, dimensions, time
    y = np.array(y) # Shape: samples, dimensions

    return X, y


def preprocessing(args):
    """
    Preprocess the clinical dataset.
    """

    ### 1. Load data ###
    clinical_df = read_clinical_df(os.getcwd() + CLINICAL_DATA_PATH)
    dtype_dict = read_clinical_dtype_dict(os.getcwd() + CLINICAL_DTYPES_PATH)

    # Checkbox column names in dtype_df do not correspond 1-on-1 with clinical data
    checkbox_dict = {
        'recru_hdu_admission_reason' : ['recru_resp', 'recru_circu', 'recru_neuro',
                                        'recru_nurse', 'recru_unclear', 'recru_hdu_other'],
        'recru_medication_specfy' : clinical_df.columns[clinical_df.columns.str.startswith('recru_medication_specfy___')].to_list(),
        'dly_new_drug' : clinical_df.columns[clinical_df.columns.str.startswith('dly_new_drug___')].to_list(),
    }

    dtype_dict['checkbox'] = [l for s in checkbox_dict.values() for l in s]

    ### 2. Clean data ###
    dtype_df = split_per_dtype(clinical_df, dtype_dict) # Split data per datatype

    if args.threshold:
        dtype_df = remove_columns(dtype_df, args.threshold, args) # Remove columns if too many missing values

    dtype_df = split_text_data(dtype_df) # Split text data further

    # Remove CRT and AVPU columns
    filter_ = dtype_df['radio'].filter(['recru_avpu_score', 'dly_avpu'])
    dtype_df['radio'] = dtype_df['radio'].drop(filter_, axis=1)
    filter_ = dtype_df['num'].filter(['recru_cr_time_result', 'dly_crt'])
    dtype_df['num'] = dtype_df['num'].drop(filter_, axis=1)

    # Add hour of the day to the data
    dtype_df['num'].insert(0, 'dly_hour', pd.to_datetime(dtype_df['dates']['dly_time'], format='mixed').round('h').dt.hour)

    # Turn categorical columns to one-hot encoding
    dtype_df['radio'] = one_hot_encoding(dtype_df['radio'])

    numerical_dtypes = ['radio', 'yesno', 'checkbox', 'calc', 'numerical']
    string_dtypes = ['text', 'record_id', 'dates']
    numerical_data = np.concatenate(([v for k, v in dtype_df.items() if k in numerical_dtypes]), axis=1)
    data = normalize(numerical_data, axis=0) # Normalize data

    ### 3. Perform Principal component analysis ###
    if args.pca:
        data = perform_PCA(data, visualize=False)

    ### 4. Apply sliding window ###
    outputs = preprocess_crt_avpu(os.getcwd() + CLINICAL_DATA_PATH)
    datetimes = dtype_df['dates']['dly_time']
    admission_dates = {k : v['recru_hospital_admission_date'].iloc[0] for \
                       k, v in clinical_df.groupby('record_id')}

    X, y = sliding_window_backward(data,
                                   outputs,
                                   datetimes,
                                   admission_dates,
                                   dtype_df['record_id'].to_frame(),
                                   args.window_size)
    
    if args.verbosity:
        print("\n=== Data statistics ===")
        print(f"Input shape: {X.shape}\noutput shape: {y.shape}\n")

    return X, y


def main(args):
    """
    Check if dataset already exists, otherwise start preprocessing process.
    NOTE: Loading dataset only requires filename, not path.
    """

    if args.saved_dataset and os.path.exists(f"{os.getcwd()}/DATA/Datasets/{args.filename}"):
        print("Loading dataset...")
        clinical_dataset = torch.load(f"{os.getcwd()}/DATA/Datasets/{args.filename}")

    else:
        print("Start propressing data...")
        X, y = preprocessing(args)
        clinical_dataset = IMPALA_Dataset(X, y)
        torch.save(clinical_dataset, f"{os.getcwd()}/DATA/Datasets/{args.filename}")

    train_dataloader, val_dataloader, test_dataloader = \
        create_dataloaders(clinical_dataset, batch_size=args.batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", action="store_true",
                        help="if set, print information regarding the data")
    parser.add_argument("--pca", action="store_true",
                        help="if set, prinicipal component analysis will be performed")
    parser.add_argument("--saved_dataset", action="store_true",
                        help="if set, use saved preprocessed dataset (if file exists)")
    parser.add_argument("--filename", action="store", type=str,
                        help="filename of dataset to be saved or loaded")
    parser.add_argument("-w", "--window_size", type=int, required=True, action="store",
                        help="set the size of the sliding windows (required)")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, action="store",
                        help="set the batch size of the dataloaders (required)")
    parser.add_argument("-s", "--seed", type=int, required=True, action="store",
                        help="set the seed of the dataloaders (required)")
    parser.add_argument("-t", "--threshold", type=float, action="store",
                        help="remove columns where missing value ratio exceeds the threshold")
    
    args = parser.parse_args()

    train_dataloader, val_dataloader, test_dataloader = main(args)
