import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from utils import pickle_out, pickle_in, get_root, is_deprived


#modality = 'DTI'
modalities = ['DTI', 'MEMPRAGE', 'BRAIN']
tools = ['vuno', 'freesurfer', 'fastsurfer', 'samseg']
regex = get_root('data', 'radiomics_features', '{}_{}', '{}', '*.pkl')
Y_dict = is_deprived
subjects = list(set([s.split(os.sep)[-2] for s in glob(regex.format(modalities[0], tools[0], '*'))]))
subjects.sort()


# I did not use this list itself, I just hardcoded a diagnostics check
included_features = ['diagnostics',
                     'original_shape',
                     'original_firstorder',
                     'original_glcm',
                     'original_glrlm',
                     'original_glszm',
                     'original_glszm',
                     'original_ngtdm']
# we will train a model for each tool
for modality in modalities:
    for tool in tools:
        data = pd.DataFrame()
        for subject in tqdm(subjects):
            # gather radiomics files for each time point
            features_per_tp = {}
            for radiomics_file in glob(regex.format(modality, tool, subject)):
                time_point = os.path.basename(radiomics_file).split(os.extsep)[0]
                time_point = datetime.strptime(time_point, '%Y%m%d_%H%M%S')
                time_point = time_point.timestamp()
                features_per_tp[time_point] = pickle_in(radiomics_file)
            # exclude patients with incomplete data
            if len(features_per_tp) < 4:
                print(subject, 'excluded for missing at least one timepoint')
                continue
            #print(len(features_per_tp))
            continue
            # order keys based chronologically
            time_points = list(features_per_tp)
            time_points.sort()

            # I have attempted speeding up the for-loop below, but it gets messy quite quickly
            # fts = [[list(features_per_tp[tp][label]) for label in features_per_tp[tp]] for tp in time_points]
            # included_fts = [ft for ft in list(set(flatten(fts))) if not ft.split('_')[0] == 'diagnostics']
            fields = [('is_deprived', Y_dict[subject])]
            for i, tp in enumerate(time_points):
                for label in features_per_tp[tp]:
                    fts = features_per_tp[tp][label]
                    included_fts = [ft for ft in fts if not ft.split('_')[0] == 'diagnostics']
                    fields = fields + [(str(i + 1) + '_' + str(label) + '_' + ft, float(features_per_tp[tp][label][ft])) for
                                    ft in included_fts]
            fields = pd.DataFrame(dict(fields), index=[subject])
            data = data.append(fields)
        pickle_out(data, 'radiomic_dataframe_' + modality + '_' + tool + '.pkl')
