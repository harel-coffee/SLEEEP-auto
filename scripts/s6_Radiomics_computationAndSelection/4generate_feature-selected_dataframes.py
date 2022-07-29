import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from utils import pickle_out, pickle_in, get_root, is_deprived, freeSurfer_lut
import numpy as np
from collections import OrderedDict

'''
# Converts individual radiomics results to Wisconsin breast cancer DataFrame format
'''

'''
#   THIS CODE HAS BEEN DEPRICATED
#   It computed absolute difference in features BEFORE and AFTER sleep deprivation.
#   It has been replaced by the new COMP_ADV_FT, which computes the coefficient of linear regression.
 
def comp_adv_ft(feature, anat_label):
    # compute feature-selected features: absolute difference between timepoints t1 and t2
    fpt = {} #  read as  Feature Per Timepoint
    fpt_c = [True] * 4
    for i in range(4):
        try:
            fpt[i] = features_per_tp[time_points[i]][anat_label][feature]
        except:
            fpt_c[i] = False

    # Sometimes a single radiomics feature may not be available.
    # We dont want to exclude immediately. Thus we compute average if available, and else we don't.
    # The logic here is that we have two days, one before potential deprivation (A) and one after (B)
    # Ideally, each our measurement for each day consists of two measurements:
    #   A = (FeaturePerTimePoint_0 + FeaturePerTimePoint_1) / 2
    # But if one of these values does not exist we can work with just one.
    #
    # We return the difference between time point A and B.

    if fpt_c[0] and fpt_c[1]:
        a = (fpt[0] + fpt[1])/2
    elif fpt_c[0]:
        a = fpt_c[0]
    elif fpt_c[1]:
        a = fpt_c[1]
    else:
        return np.nan

    if fpt_c[2] and fpt_c[3]:
        b = (fpt[2] + fpt[3])/2
    elif fpt_c[2]:
        b = fpt_c[2]
    elif fpt_c[3]:
        b = fpt_c[3]
    else:
        return np.nan

    result = a - b
    return np.nan_to_num(np.real(result))
'''


def comp_adv_ft(features_per_tp, feature, anat_label):
    # compute feature-selected feature: coefficient of logistic regression
    x, y = [], []
    for time_point, features in features_per_tp.items():
        x.append(time_point)
        y.append(features[anat_label][feature])
    return np.polyfit(x, y, 1)[0]


def fpath2timestamp(radiomics_path: str) -> float:
    """
        Converts a file path as string to float time stamp
    """
    # ...urfer_DTI\\001\\20180731_093427.pkl'
    file_name = os.path.basename(radiomics_path).split(os.extsep)[0]
    # '20001231_163427'
    time_string = datetime.strptime(file_name, '%Y%m%d_%H%M%S')
    # datetime.datetime(2000, 12, 31, 16, 34, 27)
    time_stamp = time_string.timestamp()
    # 978276867.0
    return time_stamp


def labels_in_all(features_per_time_point):
    """
        Return a subset of all anatomical labels which are present across segmentation tools.
    """
    # Retrieve the intersection of the four label NUMBER sets
    t11, t12, t21, t22 = [set(features_per_time_point[tp]) for tp in list(features_per_time_point)]
    label_number_set = list(t11.intersection(t12.intersection(t21.intersection(t22))))

    # To enable sorting based on anatomical label NAME, we briefly add this information to the list
    lns_with_names = [('-'.join(freeSurfer_lut(ln, 'name').split('-')[1:]), ln) for ln in label_number_set]
    lns_with_names.sort()

    # Finally, we discard the name information, to return only label NUMBERS
    label_number_set = [ln for _, ln in lns_with_names]
    return label_number_set


class ErrorHandler(dict):
    """
        This error handler can take error types as KEY, and instances of this error as VALUE to collect errors.
        Once errors are collected, errors are handled using the HANDLE() method, iteratively, in a manner
        specific to each error type.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def append(self, key, value):
        if key not in self.keys():
            self[key] = [value]
        else:
            self[key].append(value)

    def handle(self):
        if 'missing_scan' in self.keys():
            for sub in self['missing_scan']:
                self.subjects.remove(sub)
            print('Subjects excluded for missing at least one timepoint:', self['missing_scan'])
        if 'nans_found' in self.keys():
            print('Nan values found in columns:', self['nans_found'])


def generate_feature-selected_dataframe(subject_list: list, tool='freesurfer', modality='MEMPRAGE') -> pd.DataFrame:
    error_handler = ErrorHandler(subjects=subject_list)
    # Allocate the dataframe that we will store to
    results_data_frame = pd.DataFrame()
    # Create a dict that we can use to store errors in:
    # Add a row to the dataframe for each subject
    for subject in tqdm(subjects):
        # gather radiomics files for each time point of this subject
        features_per_tp = OrderedDict()
        # iterate chronologically over each radiomics file
        for radiomics_file in sorted(glob(regex.format(tool, modality, subject))):
            features_per_tp[fpath2timestamp(radiomics_file)] = pickle_in(radiomics_file)
        # exclude patients with incomplete data
        if len(features_per_tp) < 4:
            error_handler.append(key='missing_scan', value=subject)
            continue

        # Now that we have loaded the content of the radiomics files for this subject,
        # we should 1) make a selection as to what we would like to retain
        #           2) compute feature-selected radiomics features

        # We make a selection of the included anatomical labels (e.g. exluding skull etc.)
        selected_anatomical_labels = []
        for al in labels_in_all(features_per_tp):
            if any(i in freeSurfer_lut(al, 'name') for i in included_anatomy):
                selected_anatomical_labels.append(al)

        # We create the first instance of a list that we will append relevant features to
        fields = [('is_deprived', is_deprived[subject])]

        for label in selected_anatomical_labels:
            # We only include features from the selected included_features
            fts = []
            # The line below iterates over all feature names. These are the same for all
            # timepoints, thus we just pick the first using 'next(iter(dict))'
            for ft in features_per_tp[next(iter(features_per_tp))][label]:
                if ft.split('_')[1] in included_features:
                    fts.append(ft)

            # Finally we add everything we've found to the list...
            fields = fields + [(str(label) + '_' + ft, comp_adv_ft(features_per_tp, ft, label)) for ft in fts]
        # ...and generate a DataFrame from this list
        fields = pd.DataFrame(dict(fields), index=[subject])
        results_data_frame = results_data_frame.append(fields)

    # fill nans if at least half of subjects has the value
    for column in results_data_frame.columns:
        if any([np.isnan(el) for el in results_data_frame[column]]):
            error_handler.append(key='nans_found', value=column)

    # Handle errors
    error_handler.handle()
    return results_data_frame


if __name__ == '__main__':
    # The radiomics dataset to use to generate features from
    modality = 'DTI'
    # Expression for finding subjects and radiomics files
    regex = get_root('data', 'radiomics_features', '{}_{}', '{}', '*.pkl')

    # Selection criteria for included features
    included_features = ['firstorder', 'shape']

    included_anatomy = ['Cerebral-White-Matter', 'Cerebral-Cortex',
                        'Thalamus-Proper', 'Caudate', 'Putamen', 'Pallidum',
                        'Hippocampus', 'Amygdala', 'Accumbens-area', 'VentralDC']

    # A string you can add to the saved file name
    file_descriptor = ''

    # We will create a dataframe for each tool
    # If you are real cool you can replace the line below with
    subjects = list(set([s.split(os.sep)[-2] for s in glob(regex.format('*', modality, '*'))]))
    subjects.sort()
    for tool in ['vuno', 'freesurfer', 'fastsurfer', 'samseg']:
        data_frame = generate_feature-selected_dataframe(subjects, tool, modality)
        file_name = '_'.join(['feature-selected_radiomic_dataframe', modality, tool + '-' + file_descriptor + '.pkl'])
        pickle_out(data_frame, file_name)
        print('Successfully created:', file_name)





