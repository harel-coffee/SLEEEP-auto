from glob import glob
import os

# This code moves each file from \sleeep\rawmask
# to \newdir simply to change the original directory format to that of Vuno

for file in glob(r"/home/mgpoirot/lood_storage/divi/Projects/depredict/cohorten/sleeep/volumes/*/*/mri/"):
    if not os.listdir(file):
        os.rmdir(file)
    print('removed', file)

# srs = r"L:\basic\divi\Projects\depredict\cohorten\sleeep\rawmask"
# des = r"L:\basic\divi\Projects\depredict\cohorten\sleeep\newdir"
files_remaining = lambda srs: glob(os.path.join(srs, '*.nii.gz'))

while files_remaining():
    print(len(files_remaining()), 'files remaining')

    pt_id = files_remaining()[0].split(os.sep)[-1][:3]  # 001
    print('working on', pt_id)

    files = glob(os.path.join(srs, pt_id + '*.nii.gz'))
    print('found', len(files), 'files')

    dest = os.path.join(des, pt_id)
    for file in files:
        new_loc = os.path.join(des, pt_id, file.split(os.sep)[-1])
        os.replace(file, new_loc)
        print('moved\n-', file, 'to\n-', new_loc)
