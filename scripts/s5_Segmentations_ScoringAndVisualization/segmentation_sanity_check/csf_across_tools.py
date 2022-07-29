import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from matplotlib import pyplot as plt
import nibabel as nib
from utils import get_root, freeSurfer_lut
import numpy as np

regex = get_root('data', 'segs_{}', '{}', '{}')

tools = ['freesurfer', 'fastsurfer', 'vuno', 'samseg']


for i, tool in enumerate(tools):
    vol = nib.load(regex.format(tool, sub, tp)).get_fdata()
    plt.subplot(2, 2, i+1)
    plt.imshow(np.rot90(vol[88] == freeSurfer_lut('CSF', 'label')), cmap='gray')
    #plt.imshow(vol[:, 128, :] == freeSurfer_lut('CSF', 'label'), cmap='gray')
    #plt.imshow(vol[:, :, 128] == freeSurfer_lut('CSF', 'label'), cmap='gray')
    plt.title(tool)
    plt.axis('off')
plt.suptitle('CSF segmentation across tools')
plt.savefig('CSF segmentation across tools')
plt.show()