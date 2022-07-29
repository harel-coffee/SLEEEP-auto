'''
 VUNO_relabel.py: change labeling and conform affine transform of Vuno segmentation data to FreeSurfer format
 March 25, 2021 
 M.G.Poirot (m.g.poirot@amsterdamumc.nl)
'''

import pickle 
from glob import glob
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from utils import pickle_out, pickle_in

# in this script we do two things:
# 1. We convert the Vuno labeling scheme to FreeSurfer using a Lookup Table (LUT)
vuno_path = r'/data/projects/depredict/sleeep/segs_vuno_old_labeling/*/*.nii.gz' 
vuno_files = glob(vuno_path)
lut_path = r'/data/projects/depredict/sleeep/scripts/vuno_lut/vuno_lut.pkl'
lut = pickle_in(lut_path)

# 2. We steal the SAMSEG affine transform such that Vuno agrees with the rest.
samseg_path = r'/data/projects/depredict/sleeep/segs_samseg/*/*.nii.gz' 
samseg_files = glob(samseg_path)

# Store the result
output_path = r'/data/projects/depredict/sleeep/segs_vuno'

# For each Vuno-SAMSEG-file-pair:..
for v_f, s_f in tqdm(zip(vuno_files, samseg_files)):
	# steal samsegs affine
	samseg_affine = nib.load(s_f).affine

	# load original Vuno image data
	vol = nib.load(v_f).get_fdata()
	vol2 = np.zeros(vol.shape)

	# key-wise application of the LUT to Vuno image data
	for key, item in lut.items():
		vol2[vol == key] = item

	# Construct target file path for storage
	o_f = os.path.join(output_path, *v_f.split(os.sep)[-2:])
	if not os.path.isdir(os.path.dirname(o_f)):
		os.mkdir(os.path.dirname(o_f))

	# Construct and save Nifti image
	nii = nib.Nifti1Image(vol2, samseg_affine)
	nib.save(nii, o_f)
