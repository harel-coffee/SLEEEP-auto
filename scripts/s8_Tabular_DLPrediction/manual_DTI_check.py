"""
    ## REPRODUCIBILITY CHECK OF LITERATURE

    In this script I tried to replicated results by Elvsashagen et al. on
    reduced mean fractional anisotropy after a night of sleep deprivation.

    # Conclusions are:
    I found that the linear coefficient of the deprived group was actually positive (+3500),
    and significantly so compared to the non deprived group (-900) with a t-statistic of 3.05
    and p-value of 0.003

    July 2021 - M.G.Poirot
"""
import os 
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd
from utils import pickle_in, get_root
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

df_path = get_root('data', 'radiomics_dataframes', 'radiomic_dataframe_DTI_freesurfer.pkl')
df = pickle_in(df_path)
# freeSurfer_lut(2, 'name')
# 'Left-Cerebral-White-Matter'
# freeSurfer_lut(41, 'name')
# 'Right-Cerebral-White-Matter'
df = df.sort_values('is_deprived')

hemispheres = ['2', '41']
res = {'deprived': [], 'nondeprived': []}
x_labels = [0.0,11.0,23.6,31.1]
plt.subplot(1,2,1)
plt.title('non-deprived')
plt.ylabel('Mean fractional anisotropy across voxels $[10^{-3} mm^{2}/s]$')
plt.xlabel('Time from start in hours') 

for index, row in df.iterrows():
    query = '{}_{}_original_firstorder_Mean'
    if row['is_deprived']:
        c = 'r'
        plt.subplot(1,2,2)
        plt.title('deprived')
    else:
        c = 'g'



    for hemisphere in hemispheres:
        line = []
        for time_point in range(1, 5):
            line.append(row[query.format(str(time_point), hemisphere)])
        
        if not np.sum([i < 0.3 for i in line]):
            plt.plot(x_labels, line, c=c)
            coeff = np.polyfit(line, x_labels, 1)[0]
            if coeff > 30000:
                continue
            if row['is_deprived']:
                res['deprived'].append(coeff)
            else:
                res['nondeprived'].append(coeff)

plt.xlabel('Time from start in hours') 
plt.show()
bins = np.linspace(-10000, 40000, 10)
plt.hist(res['nondeprived'], bins=bins, histtype='step', linewidth=3, color='g')
plt.hist(res['deprived'], bins=bins,  histtype='step', linewidth=3, color='r')
plt.show()
print(ttest_ind(res['deprived'], res['nondeprived']))

    


deprived_before = df['1_2_original_firstorder_Mean'][df['is_deprived']]
deprived_before = df['1_2_original_firstorder_Mean'][df['is_deprived']]
deprived_before = df['1_2_original_firstorder_Mean'][df['is_deprived']]

deprived_after = df['4_2_original_firstorder_Mean'][df['is_deprived']]


nondeprived_before = df['1_2_original_firstorder_Mean'][~df['is_deprived']]
nondeprived_after = df['4_2_original_firstorder_Mean'][~df['is_deprived']]

plt.figure()
for i, (before, after) in enumerate(zip(list(deprived_before) + list(nondeprived_before),
                                      list(deprived_after) + list(nondeprived_after))):
    print(before, after)
    if before < 0.1 or after <0.1:
        continue
    if i <= len(deprived_after):
        c = 'r'
        print(i)
    else:
        c= 'b'
    after /= before
    before = 1
    plt.plot([before, after], c=c)
plt.show()

deprived_after = deprived_after[deprived_before > 0.1]
deprived_before = deprived_before[deprived_before > 0.1]
rel_d_d = deprived_after.divide(deprived_before)
np.mean(rel_d_d)

nondeprived_after = nondeprived_after[nondeprived_before > 0.1]
nondeprived_before = nondeprived_before[nondeprived_before > 0.1]
rel_d_nd = nondeprived_after.divide(nondeprived_before)
np.mean(rel_d_nd)

