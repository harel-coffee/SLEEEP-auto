from glob import glob
import nibabel as nib
import os
import sys

regex = r'/data/projects/depredict/sleeep/*_{}/0*/20*{}'
DIRS = ['MEMPRAGE', 'freesurfer', 'fastsurfer']


# DIRS = ['MEMPRAGE', 'freesurfer', 'fastsurfer', 'samseg', 'vuno']
# DIRS = ['*']

def unzip_file(file_path: str, mode: list) -> None:
    newfile = file_path.split(os.extsep)[0] + mode[1]
    nib.save(nib.load(file_path), newfile)
    if os.path.isfile(newfile):
        os.remove(file_path)
    print(file_path, '->', newfile)
    return


def unzip_bulk(mode: list) -> None:
    files = []
    for dir in DIRS:
        print(dir, len(glob(regex.format(dir, mode[0]))))
        files.extend(glob(regex.format(dir, mode[0])))

    if not len(files):
        print('       - NO FILES FOUND TO [UN]ZIP :( -')
        return

    for i, file in enumerate(files):
        unzip_file(file, mode)
        print(i, file)
    return


if __name__ == '__main__':
    print("\n        - WELCOME TO [UN]ZIPPER.PY! -")

    # Operation mode. Read as "convert from mode[0] to mode[1]"
    # Unzip is default. Is reverse for zipping.
    mode = ['.nii.gz', '.nii']

    # Print DIRS content
    for dir in DIRS:
        print(dir.ljust(15), *[str(len(glob(regex.format(dir, m)))) + m for m in mode])

    # Gather input about what we are going to do
    if len(sys.argv) == 2:
        response = sys.argv[1]
    else:
        response = input("What do you want to do [zip]/unzip? ")

    # Set the mode accordingly
    if response == 'unzip':
        print("          - LET'S START UNZIPPING! -")
    elif response == 'zip':
        mode.reverse()
        print("           - LET'S START ZIPPING! -")
    else:
        raise ValueError("Invalid input. Options are: [zip]/unzip")

    # Perform unzipping accorting to set properties
    unzip_bulk(mode)
