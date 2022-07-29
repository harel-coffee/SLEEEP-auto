import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
from utils import pickle_in, get_root, freeSurfer_lut, fisher_mean, Logger

def col2anat(column):
	return int(column.split('_')[1])


def col2radft(column):
	return '_'.join(column.split('_')[2:])

logger = Logger(storage_path=get_root('documents', 'graphics', 'barchart', 'barchart_full'))
# GLOBAL VARIABLES
for task in ['icc', 'icc2']:
    for vol in ['DTI', 'BRAIN']:
        results = pickle_in(get_root('scripts', 's7_Radiomics_analysisAndVisualization', 'robustness_' + task + '_' + vol + '_results.pkl'))
        columns = list(results.keys())

        # Get anatomical labels sorted
        anatomical_labels = list(set([col2anat(column) for column in columns]))
        sortlist = [freeSurfer_lut(int(i), 'name')[::-1] for i in anatomical_labels]
        anatomical_labels = [int(x) for _, x in sorted(zip(sortlist, anatomical_labels))]

        # Check if we can merge left-right results
        is_left_lut = {}
        is_right_list = []
        for anatomical_label in anatomical_labels:
            label_name = freeSurfer_lut(int(anatomical_label), 'name')
            prefix = label_name.split('-')[0]
            if prefix == 'Left':
                label_name_right = 'Right' + label_name[4:]
                label_right = freeSurfer_lut(label_name_right, 'label')
                is_right_list.append(label_right)
                is_left_lut[anatomical_label] = label_right
                try:
                    anatomical_labels.remove(label_right)
                except:
                    pass
        # The first type of lookup tabel (LUT) is to go from index to Anatomical Label
        # The AL-LUT is used to sort the fields alphabetically.
        # While we are on it, we will also create an inverse, the LA-LUT.
        # The LA-LUT is only used for the minor task of generating y-axis labels.
        al_lut, la_lut = {}, {}
        for i, al in enumerate(anatomical_labels):
            al_lut[al] = i
            la_lut[int(i)] = al


        # The second type of LUT is to go from column number to readiomic feature.
        radiomic_features = list(set([col2radft(column) for column in columns]))
        radiomic_features.sort()
        rf_lut = {}
        for k, rf in enumerate(radiomic_features):
            rf_lut[rf] = k


        radiomic_feature_classes = list(set([x.split('_')[1] for x in rf_lut]))
        radiomic_feature_classes.sort()
        excl = [freeSurfer_lut('Left-choroid-plexus', 'label'),
                freeSurfer_lut('Left-Inf-Lat-Vent', 'label'),
                freeSurfer_lut('Right-choroid-plexus', 'label'),
                freeSurfer_lut('Right-Inf-Lat-Vent', 'label'),
                freeSurfer_lut('CSF', 'label')]


        for rfc in radiomic_feature_classes:
            innies = [x for x in columns if rfc in x]  # included on basis of Radiomic Feature class
            innies = [x for x in innies if int(x.split('_')[1]) not in excl]  # Included on bases of ROI
            if task == 'icc':
                mean_iccs = [results[x]['ICC'][2] for x in innies]
                logger(','.join((task, vol, rfc, str(mean_iccs), str(fisher_mean(mean_iccs)))))
            elif task == 'icc2':
                mean_iccs = [results[x] for x in innies]
                logger(','.join((task, vol, rfc, str(mean_iccs), str(fisher_mean(mean_iccs)))))
            else:
                mean_rmaes = [results[x] for x in innies if not results[x] == np.nan]
                logger(','.join((task, vol, rfc, str(np.mean(mean_rmaes), str(mean_rmaes)))))
