import pandas as pd
import numpy as np
from collections import defaultdict
import pyarrow.parquet as pq
import pickle


def filter_range_column(column: pd.Series, lower_range: int, upper_range: int):
    """ MAX's Code
    Replace values that are outside the provided range with NaN values
    :param column: Column name
    :param lower_range: Lower range to filter
    :param upper_range: Upper range to filter
    :return: Filtered, modified dataframe
    """

    column = column.fillna(-1)
    return np.where(
        (column >= lower_range) & (column <= upper_range),
        column,
        np.nan
    )


def remove_biologically_impossible(df: pd.DataFrame):
    """
    Removes biologically impossible values from the vital signs, ranges are provided by Job
    :param df: Vital signs dataframe
    :return: Modified vital signs dataframe
    """
    biological_ranges = [
        {
            'column_name': 'ECGHR',
            'lower_range': 0,
            'upper_range': 250
        },
        {
            'column_name': 'SPO2HR',
            'lower_range': 0,
            'upper_range': 250
        },
        {
            'column_name': 'ECGRR',
            'lower_range': 0,
            'upper_range': 140
        },
        {
            'column_name': 'SPO2',
            'lower_range': 0,
            'upper_range': 100
        },
        {
             'column_name': 'NIBP_lower',
             'lower_range': 10,
             'upper_range': 100
        },
        {
            'column_name': 'NIBP_mean',
            'lower_range': 10,
            'upper_range': 150
        },
        {
             'column_name': 'NIBP_upper',
             'lower_range': 15,
             'upper_range': 200
        }
    ]
    _df = df.copy()

    for biological_range in biological_ranges:
        column_name = biological_range['column_name']
        lower_range = biological_range['lower_range']
        upper_range = biological_range['upper_range']

        n_outliers = len(_df[_df[column_name] < lower_range])
        n_outliers += len(_df[_df[column_name] > upper_range])
        outlier_percentage = (n_outliers / len(_df)) * 100
        # print(f'{column_name}: {n_outliers} outliers ({outlier_percentage:.2f}%)')

        _df[column_name] = filter_range_column(
            column=_df[column_name],
            lower_range=lower_range,
            upper_range=upper_range
        )
    return _df


def clean_batch(df):
    """
    Clean batch.
    """

    # Clean data
    df = df.drop_duplicates() # Drop duplicates
    df = df[~df['patient_id'].isna()] # Unknown patients are omitted
    df = df[~df['patient_id'].str.startswith('unknown')] # Unknown patients are omitted

    # Remove biologically impossible values
    df = remove_biologically_impossible(df)

    # Remove entry if all vital signs are NaN
    valid_indices = df[['ECGHR', 'ECGRR', 'SPO2HR', 'SPO2', 'NIBP_lower',
                        'NIBP_upper', 'NIBP_mean']].dropna(how='all').index
    
    df = df.loc[valid_indices]

    return df


def read_raw_vital_signs(path: str, batch_size=10000, n_batches=None, patient_id=None,
                         valid_columns=None):
    """
    Read the raw vital sign data and convert it to a dictionary of Pandas
    DataFrames where each key is a patient_id.
    :param filename: String; path to raw vital sign file.
    :param n_batches: Int; the data is read in batches, indicate if the data
                      should stop at a certain number of batches.
    """

    raw_parquet = pq.ParquetFile(path)

    raw_patient_df = defaultdict(list)

    if not valid_columns:
        valid_columns = ['ECGHR', 'ECGRR', 'SPO2HR', 'SPO2', 'NIBP_lower', 'NIBP_upper',
                         'NIBP_mean', 'datetime', 'patient_id', 'location']
    
    for i, batch in enumerate(raw_parquet.iter_batches(batch_size=batch_size,
                                                       columns=valid_columns)):

        if n_batches and i > n_batches:
            break

        if i % 1000 == 0:
            print(f"Batch {i}")

        if patient_id and type(patient_id) == str:
            if patient_id in batch['patient_id'].tolist():
                # print(f"Patient ID: {patient_id} found in batch {i}")
                
                batch = batch.to_pandas()
                batch = batch[batch['patient_id'] == patient_id]

                # Add df to dictionary
                raw_patient_df[patient_id].append(clean_batch(batch))

            # To keep function memory efficient
            del batch
        

        elif patient_id and type(patient_id) == list:
        
            batch = batch.to_pandas()
            for p_id in patient_id:
                
                if p_id in batch['patient_id'].tolist():

                    b = batch.copy()
                    b = b[b['patient_id'] == p_id]

                    # Add df to dictionary
                    raw_patient_df[p_id].append(clean_batch(b))

                    del b

            # To keep function memory efficient
            del batch

        else:
            batch = batch.to_pandas()
            for curr_id, df in batch.groupby('patient_id'):

                # Add df to dictionary
                raw_patient_df[curr_id].append(clean_batch(df))
        
                # To keep function memory efficient
                del curr_id
                del df


            # To keep function memory efficient
            del batch


    raw_patient_df = {k : pd.concat(v, ignore_index=True) for k, v in raw_patient_df.items()}

    return raw_patient_df


def save_patient_dict(patient_dict, path):
    """
    Save the patient_dict created by read_raw_vital_signs().
    """

    with open(path, 'wb') as file:
        pickle.dump(patient_dict, file)


def load_patient_dict(path):
    """
    Load the patient_dict created by read_raw_vital_signs().
    """

    with open(path, 'rb') as file:
        patient_dict = pickle.load(file)
    
    return patient_dict
