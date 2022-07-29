import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
from matplotlib import pyplot as plt
from utils import pickle_in, pickle_out, get_root, freeSurfer_lut, fisher_mean
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})


def col2anat(column):
    return int(column.split('_')[1])


def col2radft(column):
    return '_'.join(column.split('_')[2:])


# GLOBAL VARIABLES
resdct = {'icc':
              {'DTI':
                   {'firstorder': [], 'glszm': [], 'shape': [], 'glcm': [], 'gldm': [], 'ngtdm': [], 'glrlm': []},
               'BRAIN':
                   {'firstorder': [], 'glszm': [], 'shape': [], 'glcm': [], 'gldm': [], 'ngtdm': [], 'glrlm': []}},
          'icc2':
              {'DTI':
                   {'firstorder': [], 'glszm': [], 'shape': [], 'glcm': [], 'gldm': [], 'ngtdm': [], 'glrlm': []},
               'BRAIN':
                   {'firstorder': [], 'glszm': [], 'shape': [], 'glcm': [], 'gldm': [], 'ngtdm': [], 'glrlm': []}}}
for task in ['icc', 'icc2']:
    for vol in ['DTI', 'BRAIN']:
        results = pickle_in(get_root('scripts', 's7_Radiomics_analysisAndVisualization',
                                     'robustness_' + task + '_' + vol + '_results.pkl'))
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
                anatomical_labels.remove(label_right)
        is_left_list = anatomical_labels
        # The first type of lookup tabel (LUT) is to go from index to Anatomical Label
        # The AL-LUT is used to sort the fields alphabetically.
        # While we are on it, we will also create an inverse, the LA-LUT.
        # The LA-LUT is only used for the minor task of generating y-axis labels.
        al_lut, la_lut = {}, {}
        for i, al in enumerate(is_left_list):
            al_lut[al] = i
            la_lut[int(i)] = al

        #
        #radiomic_feature_classes = list(set([x.split('_')[1] for x in rf_lut]))
        #radiomic_feature_classes.sort()
        radiomic_feature_classes = ['ngtdm', 'glszm', 'glrlm', 'gldm', 'glcm', 'firstorder', 'shape']

        # The second type of LUT is to go from column number to readiomic feature.
        radiomic_features = list(set([col2radft(column) for column in columns]))
        radiomic_features.sort()
        rf_lut = {}
        k = 0
        for rfc in radiomic_feature_classes:
            for rf in [rf for rf in radiomic_features if rfc in rf]:
                rf_lut[rf] = k
                k += 1



        # Next we will actually start reading the intraclass correlation coefficient (ICC) to an array.
        image_array = np.ones(
            [len(al_lut), len(rf_lut) + len(radiomic_feature_classes) - 1])  # we add six for empty rows between groups
        current_class = None
        for column in columns:
            al = col2anat(column)
            if al in is_right_list:
                continue  # we do the right ones on the left pass
            if al in is_left_lut:
                left_column = column
                right_column = '_'.join(['', str(is_left_lut[al]), *column.split('_')[2:]])
                if task == 'icc':
                    value = [results[left_column]['ICC'][2],
                             results[right_column]['ICC'][2]]
                    value = fisher_mean(value)
                elif task == 'icc2':
                    value = [results[left_column],
                             results[right_column]]
                    value = fisher_mean(value)
                elif task == 'rmae':
                    value = [results[left_column],
                             results[right_column]]
                    value = np.mean(value)
                else:
                    print('BAD TASK!', task)
                    breakpoint()
            else:
                value = results[column]
                if task == 'icc':
                    value = value['ICC'][2]
            rf = col2radft(column)
            rf_class = [clas for clas in radiomic_feature_classes if clas in rf][0]
            addition = radiomic_feature_classes.index(rf_class)  # to account for gaps between boxes
            print(task.ljust(6), vol.ljust(7), column.ljust(70), str(al_lut[al]).ljust(4), str(rf_lut[rf]).ljust(5), value)
            image_array[al_lut[al], rf_lut[rf] + addition] = value
            resdct[task][vol][rf_class].append(value)

        # fig = plt.figure()
        fig, ax = plt.subplots(figsize=(20, 10))
        img = ax.imshow(image_array)

        radiomic_features_with_whitespaces = []
        memory = None
        for rfc in radiomic_feature_classes:
            for rf in [rf for rf in radiomic_features if rfc in rf]:
                group = rf.split('_')[1]
                if not memory == group:
                    if memory:
                        radiomic_features_with_whitespaces.append('')
                    memory = group
                radiomic_features_with_whitespaces.append(rf)

        rfs_short = [' '.join(rf.split('_')[1:3]) for rf in radiomic_features_with_whitespaces]

        memory = None
        for i, rf_short in enumerate(rfs_short):
            if not rf_short:
                rfs_short[i] = rf_short
                continue
            group, specifics = rf_short.split(' ')
            if memory == group:
                group = ''
            else:
                memory = group
            # group = ''
            rfs_short[i] = ' '.join([group.ljust(15), specifics])
        ax.set_xticks(list(range(len(rfs_short))))
        ax.set_xticklabels(rfs_short)

        # Y-ticks are anatomical
        ax.set_yticks(list(range(len(al_lut))))
        yticks = [freeSurfer_lut(la_lut[i], 'name') for i in range(len(la_lut))]
        for i, ytick in enumerate(yticks):
            prefix = ytick.split('-')[0]
            if prefix == 'Left':
                yticks[i] = ytick[5:]
        ax.set_yticklabels(yticks)
        plt.xticks(rotation=270)

        img.set_cmap('RdYlGn')
        clim = [0.4, 1]

        if task == 'rmae':
            clim = [0, 0.4]
            #img.set_cmap('RdYlGn_r')
            # img.set_cmap('inferno_r')


        img.set_clim(clim)
        ax.set_ylim(17.5, -.5)
        # plt.show()
        plt.savefig('robustness_' + task + '_' + vol + '.tiff')
        np.savetxt('robustness_' + task + '_' + vol + '_as_array.csv', image_array)
        plt.show()
        plt.close()

        columns = list(results.keys())
        radiomic_feature_classes = list(set([x.split('_')[1] for x in rf_lut]))

        col_cls = [x.split('_')[2] for x in columns]

#
# radiomic_feature_classes.sort()
# for task in ['icc', 'icc2']:
# 	for vol in ['DTI', 'BRAIN']:
# 		data = np.genfromtxt(get_root('scripts', 's7_Radiomics_analysisAndVisualization',
# 									  'robustness_' + task + '_' + vol + '_as_array.csv'))
# 		res_list = []
# 		for rfc in radiomic_feature_classes:
# 			for i, rfw in enumerate(radiomic_features_with_whitespaces):
# 				if rfc in rfw:
# 					res_list.append(fisher_mean(data[:, 112-i]))
# 			print(task, vol, rfc, fisher_mean(res_list))

# radiomic_feature_classes.sort()
# for task in ['icc', 'icc2']:
# 	for vol in ['DTI', 'BRAIN']:
# 		for rfc in radiomic_feature_classes:
# 			print(task, vol, rfc, fisher_mean(resdct[task][vol][rfc]))
'''
radiomic_feature_classes
## small step to barchart
results = {}
radiomic_features
import numpy as np
for modality in ['MEMPRAGE', 'DTI']:
    for metric in ['icc', 'rmae']:
		data = np.genfromtxt('robustness_{}_{}_as_array.csv'.format(metric, modality))
task = 'icc'
vol = 'MEMPRAGE'
results = pickle_in(get_root('scripts', '5radiomics', 'robustness_' + task + '_' + vol + '_results.pkl'))
'''
