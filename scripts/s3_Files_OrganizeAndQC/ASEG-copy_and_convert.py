'''
 Extract segmentation files from the FreeSurfer Working Directory
 to a folder formatted according to final (Vuno-based) template:
 "segs_freesurfer"

 The FSWD consists of:
 .../FSWD/subject_id/acquisition_id/label/file_name.MGZ
 We will go to:
 .../segs_freesurfer/subject_id/acquisition_id.nii.gz

 This script uses 'mri_convert' in CLI

 January 2021, M.G.Poirot
 Python 3
'''

# generic imports
from glob import glob
import os
from utils import get_root


# parameters
OVERWRITE = True


# define file structuring
source_dir = get_root('freesurfer_working_directory')  # path to FreeSurfer WD
target_dir = get_root('segs_freesurfer')  # where extracted files will be stored
os.makedirs(target_dir, exist_ok=OVERWRITE)

# Find all segmentations in the FreeSurfer Working Directory (FSWD)
seg_paths = glob(os.path.join(source_dir, '*', '*', 'label', 'aparc.DKTatlas+aseg.mgz'))
print('# files found:', len(seg_paths))


for progress_counter, seg_path in enumerate(seg_paths):
    print(progress_counter, 'source:', seg_path)
    subject_id, acq_id = seg_path.split(os.sep)[-4:-2]
    substrings = seg_path.split(os.sep)
    acq_id = substrings[-3]
    if acq_id == 'fsaverage':
        continue  
    subject_id = substrings[-4]
    root = os.path.join(target_dir, subject_id)
    target = os.path.join(root, acq_id + '.nii.gz')
    print(progress_counter, 'target:', target)

    if not os.path.isfile(seg_path):
        print('NOT copied: source does not exist')
        continue
    elif not os.path.isdir(root):
        os.mkdir(root)

    if os.path.isfile(target):
        if OVERWRITE:
            os.remove(target)
        else:
            print('NOT copied: target file exists already')
            continue
    os.system('mri_convert ' + seg_path + ' ' + target)
    print('\n')
