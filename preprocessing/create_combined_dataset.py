import os
import argparse
import pickle
from collections import defaultdict

from preprocess_clinical import *
from preprocessing_vital_signs import *

def read_dataset(path):
    """ Load dataset dictionary using pickle. """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def intersection(l1, l2):
    """ Find the intersection between two lists. """
    return [v1 for v1 in l1 if v1 in l2]


def combine_datasets(clinical_path, vital_sign_path, max_diff=3):
    """
    Combine the clinical and vital sign datasets by looking at which windows
    lie closest to each other.
    """

    data = {}
    c_dataset = read_dataset(clinical_path)
    v_dataset = read_dataset(vital_sign_path)
    combined_ids = intersection(list(c_dataset.keys()), list(v_dataset.keys()))
    print(f'{len(combined_ids)} matching patient IDs between the clinical and vital signs dataset')
    
    a, b = 0, 0
    for patient_id in combined_ids:
        c_data = c_dataset[patient_id]
        c_X, c_y, c_time = c_data['X'], c_data['y'], c_data['t']
        v_data = v_dataset[patient_id]
        v_X, v_y, v_time = v_data['X'], v_data['y'], v_data['t']

        # Assign which c_time indices lies closest to which v_time indices
        time_assignments = []
        if len(c_time) > len(v_time):
            for c_idx, t in enumerate(c_time):
                v_diff = [np.abs(d - t) for d in v_time]
                if np.min(v_diff) < pd.Timedelta(max_diff, unit='h'):
                    time_assignments.append((c_idx, np.argmin(v_diff)))
        else:
            for v_idx, t in enumerate(v_time):
                c_diff = [np.abs(d - t) for d in c_time]
                if np.min(c_diff) < pd.Timedelta(max_diff, unit='h'):
                    time_assignments.append([np.argmin(c_diff), v_idx])

        time_assignments = np.array(time_assignments)

        # Remove patients where the datasets' times differ too much
        if time_assignments.size == 0:
            print(f"Patient {patient_id}: clinical and vital sign data don't align")
            continue

        # Create the new data
        data[patient_id] = {
            'clinical_X': c_X[time_assignments[:, 0], :, :],
            'vital_X': v_X[time_assignments[:, 1], :, :],
            'y': c_y[time_assignments[:, 0], :]
        }

        # Count number of times ouputs don't match
        for i, j in zip(c_y[time_assignments[:, 0], :], v_y[time_assignments[:, 1], :]):
            a += 1
            if (i != j).any():
                b += 1

    print(f"{round((b/a)*100, 2)}% of outputs don't match.\nPreference of output is given to clinical data.")

    n_patients = len(data)
    n_samples = sum([a["y"].shape[0] for a in data.values()])
    # sample_length = X.shape[1]
    # n_features = X.shape[-1]

    print('=== Data statistics ===')
    print(f' No patients     : {n_patients}')
    print(f' No data samples : {n_samples}')
    # print(f' Sample length   : {sample_length}')
    # print(f' No features     : {n_features}')
    print('=======================')

    return data

def main(clinical_path, vital_sign_path):
    
    data = combine_datasets(clinical_path, vital_sign_path)

    # filename = f'CombinedDataset_w14_d4_o0.5_normalize.pkl'
    # with open('./DATA/Datasets/' + filename, 'wb') as f:
    #     pickle.dump(data, f)
    
    return data


if __name__ == "__main__":

    clinical_path = './DATA/Datasets/ClinicalDataset_w14_d4_time.pkl'
    vital_sign_path = './DATA/Datasets/VitalSignDataset_w14_d4_o0.5_normalize.pkl'

    _ = main(clinical_path, vital_sign_path)


