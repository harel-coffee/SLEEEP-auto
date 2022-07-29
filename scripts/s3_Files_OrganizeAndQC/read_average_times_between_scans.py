# read average time between scans

from glob import glob
import os
from datetime import datetime
import numpy as np

deltas = []
for sub in glob(r"D:\repositories\SLEEEP\data\vols_DTI\*"):
    times = []
    files = glob(os.path.join(sub, '*.nii.gz'))
    if len(files) < 4:
        continue
    for file in files:
        time_str = os.path.basename(file).split('_')[1].split(os.extsep)[0]
        times.append(datetime.strptime(time_str, '%H%M%S'))
    tmp = [times[1] - times[0], times[2] - times[1], times[3] - times[2]]
    deltas.append([t.seconds/3600 for t in tmp])

for i in range(3):
    mu = np.round(np.mean([j[i] for j in deltas]), 1)
    sd = np.round(np.std([j[i] for j in deltas]), 1)
    print('avg time between T{}-T{}:'.format(i+1, i), str(mu) + '+' + str(sd), 'hours.')
