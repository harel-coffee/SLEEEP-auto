# copy all files in FreeSurfer format to separate volumes folder (ASEG.mgz volumes)
# January 2021, M.G.Poirot
# Python 3

from glob import glob
import os
import subprocess

target_dir = r"/data/projects/depredict/sleeep/segs_freesurfer" 
sources = glob(r"/data/projects/depredict/sleeep/segs_samseg/*/*/seg.mgz")


print('# files found:', len(sources))

for i, source in enumerate(sources):
    print(i, 'source:', source)
    target = os.path.dirname(source) + '.nii.gz'
    print(i, 'target:', target)

    if not os.path.isfile(source):
        print('NOT copied: source does not exist')

    if os.path.isfile(target):
        print('NOT copied: target file exists already')
	continue

    os.system('mri_convert ' + source + ' ' + target)

    print('copied SUCCESSFULLY', '\n')

