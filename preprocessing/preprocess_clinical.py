import os
import argparse
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocess_outputs import preprocess_crt_avpu

CLINICAL_DTYPES_PATH = '/DATA/IMPALA_Clinical_dictionary.csv'
MISSING = np.nan


def clean_data(data: pd.DataFrame, remove_outputs: bool = True) -> pd.DataFrame:
    """
    Remove the data from the patients that withdrew their consent and remove
    duplicate rows. Also remove AVPU, CRT or any related features from the data
    as this will be the variables we want to predict (optional).
    """

    # Remove rows that have no daily time are duplicates of the row above
    data = data[-pd.isna(data['dly_time'])]
    data = data.reset_index()
    data = data.fillna(MISSING)

    # Remove patients that withdrew consent or absconded.
    # data = data[data['dis_outcome'] != 3] # Absconded
    data = data[data['dis_outcome'] != 4] # Withdrew consent

    # Remove AVPU and CRT features
    remove_cols_1 = ['recru_avpu_score', 'dly_avpu', 'recru_cr_time',
                       'recru_cr_time_result', 'dly_crt_available', 'dly_crt']

    data = data.drop(remove_cols_1, axis=1)

    # Remove other features that correlate too much
    if remove_outputs:

        # Obvious
        remove_cols_2 = ['recru_bt_coma_score_eye', 'recru_bt_coma_score_motor',
                       'recru_bt_coma_score_verbal', 'recru_bcs_total',
                       'dly_bcs_e', 'dly_bcs_m', 'dly_bcs_v', 'dly_bcs_total',
                       'recru_protration', 'recru_protration_days',
                       'recru_skinch_pinch', 'recru_skin_pinch_result']
        
        # Clinical analysis 1
        remove_cols_3 = ['dly_new_sepsis', 'dly_weak_rad_pulse', 'recru_convulsions',
                        'recru_convulsions_days', 'recru_drowsiness',
                        'recru_drowsiness_days', 'recru_not_drinking',
                        'recru_not_drinking_days', 'recru_not_eating',
                        'recru_not_eating_dys', 'recru_cough', 'recru_cough_num']
        
        # Clinical analysis 2
        remove_cols_4 = ['recru_diagnosis', 'recru_neuro', 'recru_circu',
                        'recru_admission_problem', 'recru_parasitic_malaria',
                        'recru_medication_specfy___9', 'recru_type_cie_2',
                        'recru_medication_specfy___19', 'recru_age_months',
                        'recru_blood_glucose', 'recru_malariardt',
                        'recru_blood_glucose_yn', 'recru_blood_glucose_format',
                        'recru_diagnosis_2', 'dly_condition_acc_nurse',
                        'dly_condition_acc_guardian', 'recru_num_potential_critic',
                        'recru_weight_kg']

        # Gradient analysis 1 (TCN)
        remove_cols_5 = ['recru_mother_employment', 'recru_father_employment',
                        'recru_num_of_diagnoses', 'recru_num_of_diagnosis',
                        'recru_lower_respiratory_tract', 'dly_monitor_attachd',
                        'recru_length_available', 'recru_dehydrated',
                        'recru_medication_specfy___46', 'recru_medication_specfy___56',
                        'recru_medication_specfy___77', 'recru_other_treat',
                        'dly_temp_available', 'dly_weight_taken', 'recru_difficult_in_b_v_3',
                        'recru_fever_days', 'recru_oxygen_saturation', 'recru_pcv',
                        'dly_temp', 'recru_admitted_from', 'recru_father_health',
                        'recru_saturation_type', 'recru_respiratory_tract',
                        'recru_admission_problem_1', 'recru_jaundice',
                        'recru_poisoning_intoxication', 'recru_crackels',
                        'recru_medication_specfy___65', 'recru_medication_specfy___66',
                        'recru_medication_specfy___70', 'dly_bloodglucose']

        # Gradient analysis 2 (linear model)
        remove_cols_6 = ['recru_medication_specfy___14', 'recru_weak_radial_pulse', 'recru_father_age_in_years', 'recru_admission_problem_2', 'recru_hb', 'recru_medication_specfy___33', 'recru_other', 'dly_condition_acc_nurse', 'recru_mother_mobile_phone', 'recru_num_potential_critic', 'recru_father_employment', 'recru_edema', 'recru_medication_specfy___69', 'recru_crackels', 'recru_medication_specfy___67', 'recru_type_cie', 'recru_diagnosis', 'recru_increase_breath', 'dly_cold_periph', 'recru_mother_highest_education', 'recru_medication_specfy___1', 'recru_not_eating_dys', 'recru_medication_specfy___50', 'recru_hospital', 'recru_medication_specfy___76', 'dly_other_investigation', 'recru_medication_specfy___4', 'recru_respiratory_tract_1', 'recru_respiratory', 'recru_type_cie_2', 'recru_viral', 'recru_mother_employment', 'recru_medication_specfy___22', 'recru_iv_fluids', 'recru_stridor', 'recru_medication_specfy___70', 'recru_blood_glucose', 'recru_num_of_diagnosis', 'recru_medication_specfy___98', 'recru_bleeding', 'recru_fever_days', 'recru_sex', 'recru_lower_respiratory_tract', 'recru_neckstiffness', 'recru_malaria_micr', 'recru_parasitic_malaria', 'recru_mother_health', 'recru_blood_glucose_format', 'dly_pallor', 'recru_heart_murmur', 'recru_nitrite', 'recru_medication_specfy___19', 'recru_medication_specfy___7', 'recru_medication_specfy___51', 'recru_medication_specfy___53', 'recru_father_age_estimated', 'recru_spleen_palatable_yn', 'recru_distended_abdomen', 'recru_wheezing', 'recru_medication_specfy___30', 'recru_medication_specfy___46', 'recru_medication_specfy___71', 'dly_monitor_attachd', 'recru_chronic_condit_v_0', 'recru_miscellaneous_2', 'recru_blood_culture', 'recru_medication_specfy___75', 'dly_condition_acc_guardian', 'recru_num_of_diagnoses', 'recru_medication_specfy___11', 'recru_storage_specimen_1_2ml', 'recru_medication_specfy___65', 'recru_medication_specfy___21', 'recru_medication_specfy___18', 'recru_respiratory_rate_min', 'recru_abnormalities', 'recru_child_med_insurance', 'recru_medication_specfy___77', 'recru_platelets', 'recru_ward', 'recru_convulsions_days', 'recru_respiratory_tract', 'recru_length_available', 'recru_medication_specfy___72', 'dly_bloodglucose', 'recru_bacterial', 'recru_medication_specfy___57', 'recru_previous_admissions', 'recru_siblings_alive_num', 'recru_bld_culture_results', 'dly_new_critical1', 'dly_weight_taken', 'recru_medication_specfy___9', 'recru_trauma', 'recru_study_lactate', 'recru_blood_glucose_yn', 'recru_pallor', 'recru_diagnosis_2', 'recru_medication_specfy___34', 'recru_hdu_other', 'recru_father_highest_education', 'recru_medication_specfy___56', 'recru_diarrhoea', 'recru_malariardt', 'recru_bloodculture', 'recru_medication_specfy___40', 'recru_unclear', 'recru_hiv_status', 'recru_father_health', 'recru_pallor_history', 'recru_convulsions', 'recru_medication_specfy___78', 'recru_admission_problem', 'recru_medication_specfy___54', 'recru_leukocytes', 'recru_medication_specfy___29', 'recru_medication_specfy___64', 'recru_father_mobile_phone', 'dly_wob', 'recru_blood_culture_yn', 'recru_admission_problem_1', 'recru_medication_specfy___32', 'recru_medication_specfy___49', 'recru_medication_specfy___66', 'recru_medication_specfy___12', 'recru_grunting', 'recru_medication_specfy___68', 'recru_neuro', 'recru_admitted_from', 'recru_urine_2_10ml', 'recru_wbc']
        
        
        data = data.drop(remove_cols_2, axis=1)
        # data = data.drop(remove_cols_3, axis=1)
        # data = data.drop(remove_cols_4, axis=1)
        # data = data.drop(remove_cols_5, axis=1)
        data = data.drop(remove_cols_6, axis=1)

    return data


def load_data(file_path: str, dict_path: str) -> (pd.DataFrame, defaultdict):
    """
    Load the clinical data into a Pandas DataFrame along with dictionary with
    the data types of the clinical data features.
    """

    # Load and clean data
    data = pd.read_csv(file_path, low_memory=False)
    data = clean_data(data)

    # Load and clean data type dictionary
    df = pd.read_csv(os.getcwd() + dict_path)
    df = df[df['Variable'].str.startswith(tuple(['recru', 'dly', 'record_id']))]

    # Checkbox column names in dtype_df do not correspond 1-on-1 with clinical data
    checkbox_dict = {
        'recru_hdu_admission_reason' : ['recru_resp', 'recru_circu', 'recru_neuro',
                                        'recru_nurse', 'recru_unclear', 'recru_hdu_other'],
        'recru_medication_specfy' : data.columns[data.columns.str.startswith('recru_medication_specfy___')].to_list(),
        'dly_new_drug' : data.columns[data.columns.str.startswith('dly_new_drug___')].to_list(),
    }

    # Filter field types and possible values (choices) per row
    ddict = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():

        if row['Field Type'] in 'text':
            pass

        # If radio contains only two choices, assign it to the yesno columns
        elif row['Field Type'] == 'radio':
            row['Choices'] = np.array([t.split(', ')[0] for t in row['Choices'].split(' | ')], dtype=int)
            
            if len(row['Choices']) < 4 and 0 in row['Choices']:
                row['Field Type'] = 'yesno'
                row['Choices'] = float('NaN')
    
        elif row['Field Type'] == 'yesno':
            pass

        # Checkbox columns don't align 1-on-1 with clinical data, so add each
        # choice as a separate yesno column
        elif row['Field Type'] == 'checkbox':
            row['Choices'] = np.array([t.split(', ')[0] for t in row['Choices'].split(' | ')], dtype=int)

            if row['Variable'] in checkbox_dict:
                row['Variable'] = checkbox_dict[row['Variable']]
                row['Field Type'] = 'yesno'
                row['Choices'] = float('NaN')

                ddict[row['Field Type']]['name'].extend(row['Variable'])
                ddict[row['Field Type']]['choices'].append(row['Choices'])

            continue

        elif row['Field Type'] == 'calc':
            row['Choices'] = float('NaN')
            pass

        ddict[row['Field Type']]['name'].append(row['Variable'])
        ddict[row['Field Type']]['choices'].append(row['Choices'])

    return data, ddict


def intersection(l1: list, l2: list) -> list:
    """ Return the intersection of two lists. """
    return [v1 for v1 in l1 if v1 in l2]


def split_data_per_type(data: pd.DataFrame, ddict: defaultdict) -> dict:
    """
    Split data per data type. Replace unknown or missing values with nan.
    Extract the numerical data from the text data and discard the rest. Also
    save the record_id data and daily time.
    """

    split_data = dict()

    for key, v in ddict.items():
        valid_cols = intersection(v['name'], data.columns.to_list())

        if valid_cols != []:
            df = data[valid_cols]

            # Fill 99 in radio
            if key == 'radio':
                df = df.replace(to_replace=99, value=MISSING)
                split_data['categorical'] = df.convert_dtypes()
            
            # Fill 99 in yesno
            elif key == 'yesno':
                df = df.replace(to_replace=99, value=MISSING)
                split_data['binary'] = df.convert_dtypes()

                if 'recru_sex' in df:
                    split_data['binary']['recru_sex'][split_data['binary']['recru_sex'] == 2] = 0

            # Remove text columns that are actually text
            elif key == 'text':
                numerical_df = df.select_dtypes(include=['float64', 'int64'])
                # Add hour of the day to the numerical data
                # numerical_df.insert(0, 'dly_hour', pd.to_datetime(df['dly_time'], format='mixed').round('h').dt.hour)
                split_data['numerical'] = numerical_df.convert_dtypes()

                time_and_records = df[['record_id', 'dly_time']].reset_index(drop=True)
                time_and_records['dly_time'] = pd.to_datetime(time_and_records['dly_time'], format='mixed')

            # Add calc featuers to categorical
            elif key == 'calc':
                split_data['categorical'] = pd.concat((split_data['categorical'],
                                                       df.convert_dtypes()), axis=1)

    return split_data, time_and_records


def thresholding(data: dict, threshold: float, verbose: bool = True) -> dict:
    """ Remove features where missing value rate exceeds the threshold. """
    
    remove_cols = []

    for key, df in data.items():
        before = df.shape[1]
        data[key] = data[key].drop([c for c in df if df[c].isna().sum() / \
                                    df[c].size > threshold], axis=1)
        
        remove_cols.append((key, before - data[key].shape[1]))

    if verbose:
        print(f'Thresholding at {int(threshold*100)}% missing data removes:')
        for k, v in remove_cols:
            print(f' - {v} {k} columns/features')
        print()

    return data


def one_hot_encoding(data: pd.DataFrame) -> pd.DataFrame:
    """ Turn categorical columns into multiple binary columns. """

    remove_cols = []

    for col in data:
        data = data.join(pd.get_dummies(data[col], prefix=col, prefix_sep='-',
                                        sparse=True, dtype=float))
        remove_cols.append(col)

    data = data.drop(remove_cols, axis=1)

    return data


def normalize(data: pd.DataFrame) -> np.ndarray:
    """
    Normalize the data between 0 and 1.
    NOTE: this function takes a pd.DataFrame and returns a np.array
    """
    return MinMaxScaler().fit_transform(data)


def principal_component_analysis(data: np.ndarray) -> np.ndarray:
    """
    Perform Principal Component Analysis on the data to reduce dimensionality.
    """

    pca = PCA(n_components='mle')
    data = pca.fit_transform(data)
    return data


def sliding_window(args, data: np.ndarray, datetimes: pd.Series, output: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Apply a sliding window over the given data. If a window is too long, the first
    entries are discarded. If a window is too short it is discarded completely.
    Add overlap parameter to control how much the windows should overlap.

    :param data: NumPy Array containing the clinical data features
    :param datetimes: List containing the starting times for each window.
    :param outputs: Pandas Dataframe containing CRT and AVPU values per time.
    :param args.window_hours: Desired length of the sliding windows in hours.
    """

    X, y = [], []

    for end in datetimes[::-1]:

        # Create window
        start = end - pd.Timedelta(args.window_length, unit='h')
        idx = datetimes.between(start, end)
        window = data[idx]

        # Window size constraints
        optimal_window_size = int((args.window_length / 4) + 1)
        if window.shape[0] < optimal_window_size:
            continue
        elif window.shape[0] > optimal_window_size:
            window = window[-optimal_window_size:]
        
        # Choose corresponding output (nearest to endtime and within X hours)
        time_diff = [abs(end - t) for t in output.index]
        diff_within_range = [i for i, t in enumerate(time_diff) if t <= pd.Timedelta(args.deviation, unit='h')]
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
    sign windows ('X') and corresponding output ('y') per patient_id (key).
    """

    file_path = os.getcwd() + args.data_dir
    data, ddict = load_data(file_path, CLINICAL_DTYPES_PATH)
    output_dict = preprocess_crt_avpu(file_path)

    data, time_and_records = split_data_per_type(data, ddict)

    # Thresholding
    if args.threshold:
        data = thresholding(data, args.threshold, verbose=args.verbose)

    # One-hot Encoding
    data['categorical'] = one_hot_encoding(data['categorical'])

    # print([l for s in [v.columns.to_list() for v in data.values()] for l in s])

    # Normalizing data
    data['numerical'] = normalize(data['numerical'])

    # From multiple DataFrames to one Array
    new_data = np.concatenate([v for v in data.values()], axis=1)
    new_data = np.where(pd.isna(new_data), MISSING, new_data).astype(float)

    # Principal Component Analysis
    if args.pca:
        new_data = principal_component_analysis(new_data)

    # Sliding window
    patient_data = dict()
    i = 0
    for patient_id, df in time_and_records.groupby('record_id'):
        
        if patient_id not in output_dict:
            print(f'Patient {patient_id} not found in output data')
            continue

        X, y = sliding_window(args, new_data[df.index, :], df['dly_time'], output_dict[patient_id])

        if X.shape[0] == 0:
            i += 1
            continue

        patient_data[patient_id] = {'X': X, 'y': y}

    print(f'{i} patients discarded due to insufficient amounts of data')

    if args.verbose:
        n_patients = len(patient_data)
        n_samples = sum([a["y"].shape[0] for a in patient_data.values()])
        sample_length = X.shape[1]
        n_features = X.shape[-1]

        print('=== Data statistics ===')
        print(f' No patients     : {n_patients}')
        print(f' No data samples : {n_samples}')
        print(f' Sample length   : {sample_length}')
        print(f' No features     : {n_features}')
        print('=======================')

    return patient_data


def main(args):

    data = preprocessing(args)

    if args.results_dir:
        filename = f'ClinicalDataset_w{args.window_length}_d{args.deviation}.pkl'
        with open(os.getcwd() + args.results_dir + filename, 'wb') as f:
            pickle.dump(data, f)
    
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true',
                        help='show additional informaton')
    parser.add_argument('--data_dir', action='store', type=str, required=True,
                        help="path to data directory")
    parser.add_argument('--results_dir', action='store', type=str,
                        help='path to directory to save the preprocessed data to')
    parser.add_argument("-t", "--threshold", type=float, action="store",
                        help="remove columns where missing value ratio exceeds the threshold")
    parser.add_argument('--pca', action='store_true',
                        help='show additional informaton')

    # Sliding window arguments
    parser.add_argument('-w', '--window_length', type=int, required=True,
                        action='store', help='size of sliding window in hours')
    parser.add_argument('-d', '--deviation', type=int, required=True,
                        action='store', help='maximum output deviation (hours)')
    
    args = parser.parse_args()

    assert args.threshold > 0 and args.threshold < 1, 'threshold must be > 0 and < 1'
    
    _ = main(args)
