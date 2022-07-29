


# copy all files in FreeSurfer format to separate volumes folder (aparc.DKTatlas+aseg.deep.mgz volumes)
# March 2021, M.G.Poirot
# Python 3

from glob import glob
import os
import subprocess

target_dir = r"/data/projects/depredict/sleeep/segs_fastsurfer" 
sources = glob(r"/data/projects/depredict/sleeep/trash_can/FastSurfer/my_fastsurfer_analysis/*/*/mri/aparc.DKTatlas+aseg.deep.withCC.mgz")

print('# files found:', len(sources))

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

for i, source in enumerate(sources):
    print(i, 'source:', source)
    date = source.split(os.sep)[-3]
    sid = source.split(os.sep)[-4]
    target = os.sep + os.path.join(*source.split(os.sep)[:5], 'segs_fastsurfer', sid, date + '.nii.gz')
    print(i, 'target:', target)
    
    if not os.path.isfile(source):
        print('NOT copied: source does not exist')

    if os.path.isfile(target):	
        print('NOT copied: target file exists already')
    
    if not os.path.isdir(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    os.system('mri_convert ' + source + ' ' + target)

    print('copied SUCCESSFULLY', '\n')

