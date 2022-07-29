'''
    FreeSurfer-reverstacing.m saves the new files with an r- prefix.
    This code cleans that up.

    April, 2021 m.g.poirot@amsterdamumc.nl
'''

from glob import glob
from os import path
import os

regex = r'/data/projects/depredict/sleeep/segs_*/*/r*.nii' 

ls = glob(regex)

for l in ls:
    dirname, basename = path.dirname(l), path.basename(l)
    newname = path.join(dirname, basename[1:])
    #nib.save(nib.load(l), newname + '.gz')
    print(newname)
    print(l)
    os.remove(newname)
    os.remove(l)
    
