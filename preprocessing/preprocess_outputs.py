import pandas as pd
import numpy as np


def read_clinical_df(path):
    """ Load clinical data into a Pandas DataFrame. """
    return pd.read_csv(path, low_memory=False)


def get_outputs(df):
    """
    Extract and clean data regarding the output variables, CRT and AVPU.
    NOTE: Function assumes that each entries in the recru and dis is the same
    for the entire record_id.
    """

    df = df.copy()

    # Extract CRT and AVPU columns
    df = df[['record_id', 'recru_hospital_admission_date', 'dly_time',
             'dis_hospital_discharge', 'dis_date_of_intervw',
             'recru_cr_time_result', 'dly_crt', 'dis_cr_time',
             'recru_avpu_score', 'dly_avpu']].copy()
    
    # Remove rows that have no daily time (as those are duplicates of the row above)
    df = df[-pd.isna(df['dly_time'])]

    # Convert biological impossibility to NaN
    df[df[['recru_cr_time_result', 'dly_crt', 'dis_cr_time']] > 12] = np.nan
    df[df[['recru_cr_time_result', 'dly_crt', 'dis_cr_time']] < 0] = np.nan
    df[df[['recru_avpu_score', 'dly_avpu']] > 4] = np.nan
    df[df[['recru_avpu_score', 'dly_avpu']] < 1] = np.nan

    # Remove rows where all entries are missing
    df =  df.loc[df[['recru_cr_time_result', 'dly_crt', 'dis_cr_time',
                     'recru_avpu_score', 'dly_avpu']].dropna(how='all').index]
    
    output_dict = {}

    # Create CRT and AVPU score per timestamp:
    for record_id, patient_df in df.groupby('record_id'):

        out = patient_df.copy()

        # Daily
        out = patient_df[['dly_time', 'dly_avpu', 'dly_crt']]
        out = out.rename(columns={'dly_time' : 'time', 'dly_avpu' : 'avpu', 'dly_crt' : 'crt'})
        out = out.set_index('time')

        # Admission
        out.loc[patient_df['recru_hospital_admission_date'].iloc[0]] = \
            patient_df['recru_cr_time_result'].iloc[0]
        
        out.loc[patient_df['recru_hospital_admission_date'].iloc[0]] = \
            patient_df['recru_avpu_score'].iloc[0]

        # Discharge
        if not pd.isna(patient_df['dis_cr_time']).iloc[0]:
            if not pd.isna(patient_df['dis_hospital_discharge'].iloc[0]):
                out.loc[patient_df['dis_hospital_discharge'].iloc[0]] = \
                    patient_df['dis_cr_time'].iloc[0]
            else:
                out.loc[patient_df['dis_date_of_intervw'].iloc[0]] = \
                    patient_df['dis_cr_time'].iloc[0]

        # Sort df by time
        out.index = pd.to_datetime(out.index, format='mixed')
        out = out.sort_index()

        # Remove rows where all entries are missing
        out = out.loc[out[['avpu', 'crt']].dropna(how='any').index]

        output_dict[record_id] = out

    return output_dict


def preprocess_crt_avpu(path):
    """
    Load the clinical data, extract the CRT and AVPU and collect the data in a
    Python dictionary.
    :param path: String containing path to clinical dataset.
    """

    clinical_df = read_clinical_df(path)
    output_dict = get_outputs(clinical_df)
    return output_dict
