import nibabel as nib
from glob import glob
import os

files = glob(r"D:\repositories\SLEEEP\data\vols_DTI\*\*.nii.gz")
regex = r"D:\repositories\SLEEEP\data\vols_MEMPRAGE\{}\{}"

tolerance = 10**-7

for file in files:
    sub, fname = file.split(os.sep)[-2:]
    dti = nib.load(file).affine
    mpr = nib.load(regex.format(sub, fname)).affine
    if sum(sum((dti - mpr) > tolerance)):
        print(file)