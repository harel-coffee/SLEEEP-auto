from glob import glob
import os
import torch
import nibabel as nib
from torch import nn

# Create binary classification labels from Kodene.txt and store it in a Look-Up
#  Table (LUT) of type dict, which was later included in utils.py as 
# 'is_deprived' and subsequently included in the radiomics dataframe data/
# MGPoirot


label_lut = {}
with open('Kodene.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    sub_id, label = line[:3], line[4]
    if label == 'S':
        label_lut[sub_id] = 1
    elif label == 'D': 
        label_lut[sub_id] = 0
    else:
        print(sub_id, label, 'excluded')


# Check the lut for complete patients
fname = 'fsl_dti_FA_t{}_{}.nii.gz'
niis = glob(fname.format('*', '*'))

ok = True
for sid in label_lut:
    for i in range(1, 5):
        cur_fname = fname.format(str(i), sid)
        if not os.path.isfile(cur_fname):
            print(cur_fname, 'INCOMPLETE')
            ok = False
if ok:
    print('All correct.')
print(len(label_lut.keys()))
# Setup ML 
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

nii = nib.load(cur_fname)
print(nii.shape)