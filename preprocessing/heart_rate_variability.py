import numpy as np
from hrvanalysis import remove_outliers, remove_ectopic_beats, \
                        interpolate_nan_values, get_frequency_domain_features, \
                        get_time_domain_features, get_csi_cvi_features

"""
Frequency-domain features: 'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu',
                           'total_power', 'vlf'
Time-domain features: 'mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50', 'nni_20',
                      'pnni_20', 'rmssd', 'median_nni', 'range_nni', 'cvsd',
                      'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr'
Nonlinear-domain features: 'csi', 'cvi', 'modified_csi'
"""

def calculate_hrv(data, return_features=None, verbose=False):
    """
    Convert heart rate data to RR interval data and perform heart rate
    variability (HRV) analysis.

    :param data: Numpy array with heart rate data
    :param return_features: list of strings with features to return
    :param verbose: If True, shows information about detected outliers
    :return interpolated_nn_intervals
    :return frequency_domain_features
    :return time_domain_features
    :return nonlinear_domain_features
    """

    # Convert heart rate per minute to RR intervals in milliseconds
    with np.errstate(divide='ignore', invalid='ignore'):
        rr_intervals = (60 / data) * 1000

    # This remove outliers from signal (low_rri = 250 HR, high_rri = 30 HR)
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals,  
                                                    low_rri=240, high_rri=6000,
                                                    verbose=verbose)

    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                       interpolation_method="linear")

    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals,
                                             method="malik", verbose=verbose)

    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

    # If there are still nan values, drop them
    interpolated_nn_intervals = [x for x in interpolated_nn_intervals if x == x]

    # If all entries were nan, return immediatelly
    if interpolated_nn_intervals == []:
        return dict()

    # Get features
    frequency_domain_features = get_frequency_domain_features(interpolated_nn_intervals)
    time_domain_features = get_time_domain_features(interpolated_nn_intervals)
    nonlinear_domain_features = get_csi_cvi_features(interpolated_nn_intervals)

    if return_features: # return 1 dictionary
        return {k : v for k, v in (frequency_domain_features | time_domain_features | \
                                   nonlinear_domain_features).items() if k in return_features}

    else: # return 3 dictionaries
        return frequency_domain_features | time_domain_features | nonlinear_domain_features


def apply_hrv(data, return_features=['lfnu'], threshold=280, source='ECG'):
    """
    Calculate the heart rate variability on a list of numpy arrays.
    Threshold is 280 valid seconds, because for a reliable HRV analysis a
    minimum of 5 minutes of data is required. 5 minutes is 300 seconds, so 280
    to add a little wiggle room.
    """

    hrv_features = ['lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power',
                    'vlf', 'mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50',
                    'nni_20', 'pnni_20', 'rmssd', 'median_nni', 'range_nni',
                    'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr',
                    'csi', 'cvi', 'modified_csi']
    
    # Make sure given hrv feature is valid
    assert sum([1 for v in return_features if v not in hrv_features]) == 0, \
        "Invalid hrv feature encountered"


    hrv_data = []
    
    for window in data:

        # Check heart rate source
        if source == 'ECG':
            hr_data = window[:, 0]
        elif source == 'SPO2':
            hr_data = window[:, 2]
        else:
            raise NameError(f'{source} is an invalid heart rate source')

        # Check if window contains enough valid entries
        if np.where(~np.isnan(hr_data), 1, 0).sum() >= threshold:

            hrv_features = calculate_hrv(hr_data, return_features=return_features)

            if hrv_features:
                hrv_data.append(list(hrv_features.values()))
                continue

        hrv_data.append([-999] * len(return_features))

        
    return np.array(hrv_data)
