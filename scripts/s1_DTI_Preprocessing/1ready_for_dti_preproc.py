import os
from glob import glob
import nibabel as nib
dtidir = 'dti_working_dir'
root = r'/home/mgpoirot/lood_storage/divi/Projects/depredict/repositories/SLEEEP/data/'
dtidir = 'dti_working_dir'
dtidir = os.path.join(root, dtidir)

if not os.path.isdir(dtidir):
    os.mkdir(dtidir)

for t1_path in glob(os.path.join(root, 'vols_MEMPRAGE', '*', '*.nii.gz')):
    sub, fname = t1_path.split(os.sep)[-2:]
    
    di_path = os.path.join(root, 'dwis_DTI', sub, fname)
    if not os.path.isfile(di_path):
        continue
    else:
        t1_nii = nib.load(t1_path)
        di_nii = nib.load(di_path)
    
    target_dir = os.path.join(dtidir, sub)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    fname = fname.split(os.extsep)[0]
    
    nib.save(t1_nii, os.path.join(target_dir, fname + '_t1.nii'))
    nib.save(di_nii, os.path.join(target_dir, fname + '_di.nii'))
    
