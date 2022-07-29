# Copy all files in FreeSurfer file structure format to separate volumes folder (MEMPRAGE volumes)
# January 2021, M.G.Poirot
# Python 3

from glob import glob
import os
from shutil import copyfile

regex = r"/data/projects/depredict/sleeep/freesurfer_working_directory/*/*/MEMPRAGE.nii.gz" 

target_dir = r"/data/projects/depredict/sleeep/MEMPRAGE_volumes" 


ls = glob(regex)

print('REGEX used:   ', regex)
print('# files found:', len(ls))

for i, source in enumerate(ls):
    print(i, 'source:', source)
    substrings = source.split(os.sep)
    dname = substrings[-2]    
    fname = substrings[-3]
    root = os.path.join(target_dir, fname)
    target = os.path.join(root, dname + '.nii.gz')
    print(i, 'target:', target, '\n')

    if not os.path.isfile(source):
        print('NOT copied: source does not exist')

    if os.path.isfile(target):
        print('NOT copied: target file exists already')

    if not os.path.isdir(root):
        os.mkdir(root)

    copyfile(source, target)
    print('copied SUCCESSFULLY')
