import nibabel as nib
from glob import glob
import numpy as np
import os

root = '/home/mgpoirot/lood_storage/divi/Projects/depredict/repositories/SLEEEP/data'
target_dir = os.path.join(root, 'vols_DTI')
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
regex = 'dti_working_dir/*/r*_di.nii'

for file in glob(os.path.join(root, regex)):
    sub, fname = file.split(os.sep)[-2:]
    subdir = os.path.join(target_dir, sub)
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    fname = fname[1:-7] + '.nii.gz'
    fpath = os.path.join(subdir, fname)
    print(fpath)
    if os.path.isfile(fpath):
       continue
    nii = nib.load(file)
    nii = nib.Nifti1Image(np.nan_to_num(nii.get_fdata()), nii.affine)
    nib.save(nii, fpath)
    
