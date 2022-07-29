# I used this scipt to completely clear directories where files were missing after processing it in the first run
# It deletes everthing except ".../SUBJECTS_DIR/subjid/mri/orig.mgz" IF 'aparc.DKTatlas+aseg.mgz' is not present.
# M.G.Poirot - April 2021

from glob import glob
from os import path, remove
from shutil import rmtree

def remove_except(object_path: str, exclusion: str, do_confirm=False) -> bool:
    if path.basename(object_path) == exclusion:
        return False
    else:
        if do_confirm:
            if not input('Do you want to remove "' + object_path + '"? [yes]/no ') in 'Yesyes':
                return False
        if path.isfile(object_path):
            proper_rm = remove
        else:
            proper_rm = rmtree
        proper_rm(object_path)
        return True


subdir_regex = r'/data/projects/depredict/sleeep/freesurfer_working_directory/*/*_*'
for subdir in glob(subdir_regex):
    if not path.isfile(path.join(subdir, 'mri', 'aparc.DKTatlas+aseg.mgz')):
        for level, exception in [(('*'), 'mri'), (('mri', '*'), 'orig')]:
            targets = glob(path.join(subdir, *level))
            for target in targets:
                if remove_except(target, exception, do_confirm=False):
                    print(target, 'deleted.')
                else: 
                    print(target, 'skipped.')
