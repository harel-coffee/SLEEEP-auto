from utils import pickle_in, freeSurfer_lut
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm 

f = r'/data/projects/depredict/repositories/SLEEEP/data/radiomics_dataframes/radiomic_dataframe_BRAIN_samseg.pkl'

df = pickle_in(f)

res = []
for key_l in tqdm(df):
    try:
        tp, label_l = key_l.split('_')[:2]
        rest = key_l.split('_')[2:]
        name_l = freeSurfer_lut(int(label_l), 'name')
        if 'Left' in name_l:
            name_r = name_l.replace('Left', 'Right')
            label_r = str(freeSurfer_lut(name_r, 'label'))
            key_r = '_'.join([tp, label_r, *rest])
            r = pearsonr(df[key_l], df[key_r])[0]
            res.append(r)
    except:
        continue
print(np.nanmean(res))

# avergae left-right:
# rho = 0.9332
# individual left-right:
# rho = 0.5887

# we must conclude that results uon group level are very similar,
# however, on an individual level, they may be very different...
#
# MGPoirot Nov 2021