import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from glob import glob
from utils import get_root, freeSurfer_lut
from tqdm import tqdm
import numpy as np 

regex = get_root('data', 'segs_{}', '{}', '{}')

tools = ['freesurfer', 'fastsurfer', 'vuno', 'samseg']


def colorplane(plane: np.array):
    img = np.zeros([*plane.shape, 3])
    for u in tqdm(range(img.shape[0])):
        for v in range(img.shape[1]):
            try:
                img[u, v, :] = np.divide(freeSurfer_lut(int(plane[u, v]), 'rgb'), 255)
            except:
                pass
    return img

'''
GENERATES 3 SUBPLOTS WITH VISUALIZATION OF SEGMENTATINOS
DUMPS IT IN THIS FOLDER
'''

for file in glob(regex.format(tools[0], '*', '*.nii.gz')):
    sub, tp = file.split(os.sep)[-2:]
    file_list = []
        
    for tool in tools:
        vol = nib.load(regex.format(tool, sub, tp)).get_fdata()
        plt.figure(figsize=(16,12))
        plt.subplot(2, 2, 1)
        plt.imshow(np.rot90(colorplane(vol[88, :, :])))
        plt.subplot(2, 2, 2)
        plt.imshow(np.rot90(colorplane(vol[:, 128, :])))
        plt.subplot(2, 2, 3)
        plt.imshow(np.rot90(colorplane(vol[:, :, 128])))
        plt.suptitle(file[len(get_root()):])
        plt.tight_layout()
        plt.savefig(os.path.basename(file).split(os.extsep)[0] + '_' + tool + '.png')
        plt.close()





