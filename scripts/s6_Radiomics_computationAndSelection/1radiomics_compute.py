'''
	# Purpose
	Reads SOURCE_VOL nifti file and SEGS segmentation file to compute radiomic
	features using radiomics featureextractor. SOURCE_VOL can be either 
	MEMPRAGE or DTI.

	# PARALELLIZATION (optional)
	Computes radiomics features for each:
	- 4 Tool (TOOLS)
	- 100 anatomical areas
	- 108 features
	- 180 volumes
	Which results in about a million calculations, or aboubt 800 embarasingly parallel processes.
	Processing time is reduced by about the number of CPU kernels available (x96 for me).

	# AUTHOR
	April 2021 - M.G. Poirot (maartenpoirot@gmail.com)
'''
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from radiomics import featureextractor as fe
import nibabel as nib
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count
import progressbar as pgb
from utils import pickle_out, get_root, flatten


def get_uniques(path: str) -> list:
    '''
		From a given path to a segmentation labelmap, return a list of unique labels.
	'''
    if not os.path.isfile(path):
        # We got the path string by searching for it.
        # If this error occurs something strange is going on.
        print('FILE MISSING WARNING:', path)
        return
    try:
        label_array = nib.load(path).get_fdata()
        return [int(x) for x in np.unique(label_array)][1:]
    except:
        print('FILE READ ERROR:', path)
        return


def check_if_done(files: list, remove_tmps=False) -> list:
    '''
		Filters the files list based on OVERWRITE preferences, and file existence.
	'''

    result = []
    for file in files:
        sub, ses, seg = file
        ses = ses.split(os.extsep)[0]

        # Sometimes FreeSurfer creates fsaverage link folders, ignore these
        if ses == 'fsaverage':
            continue

        file_name = dir_format.format('radiomics_features/'+source_vol, seg, sub, ses, 'pkl')
        
        if remove_tmps and os.path.isfile(file_name + '_tmp'):
            os.remove(file_name + '_tmp')

        print_fname = file_name[len(get_root('data', 'radiomics_features')):].ljust(60)
        if not os.path.isfile(file_name):
            print(print_fname, 'NOT DONE - creating new')
        else:
            if OVERWRITE:
                print(print_fname, 'DONE - overwriting')
            else:
                print(print_fname, 'DONE - skipping')
                continue
        result.append(file)
    return result


def multiply_for_each_tool(files):
    results = []
    for file in files:
        for tool in tools:
            results.append(file + [tool])
    return results


# compute
def compute_radiomics(file: tuple):
    # Identifying our target
    sub, ses, seg = file
    ses = ses.split(os.extsep)[0]

    # create the file structure to which to save our results
    file_name = dir_format.format('radiomics_features/'+source_vol, seg, sub, ses, 'pkl')
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)
    working_on_file = file_name + '_tmp'
    if os.path.exists(working_on_file):
        return
    else:
        open(working_on_file, 'a').close()


    # Error logging: Define name and clean up old error logs
    error_file_name = dir_format.format('radiomics_features/'+source_vol, seg, sub, ses + '_errors-{}', 'txt')
    error_files = glob(error_file_name.format('*'))
    for error_file in error_files:
        os.remove(error_file)

    # Specify the label (segmentation) and image (MEMPRAGE or DTI) file paths
    label_path = dir_format.format('segs', seg, sub, ses, 'nii.gz')
    image_path = dir_format.format('vols', source_vol, sub, ses, 'nii.gz')

    # Get a list of unique labels in the segmentation so we can iterate over each unique label next
    unique_labels = get_uniques(label_path)

    results = {}
    error_list = []
    bar = pgb.ProgressBar(maxval=len(unique_labels),
                          widgets=[os.path.join(source_vol + '_' + seg, sub, ses + '.pkl'),
                                   pgb.Counter(), '/', str(len(unique_labels)),
                                   pgb.Bar('█', ' |', '| '), pgb.AdaptiveETA(), '\n'])
    bar.start()
    for i, ul in enumerate(unique_labels):
        bar.update(i + 1)
        occurrence_of_ul = np.sum(nib.load(label_path).get_fdata() == ul)
        # With checking if the label is present in the volume we try to catch some errors
        if occurrence_of_ul <= 1:
            error_list.append((ul, 'occurrence:' + str(occurrence_of_ul)))
            results[ul] = {}
            continue
        # Clear the extractor
        extractor = fe.RadiomicsFeatureExtractor(**params)
        # We store an error notice and leave the field empty
        try:
            result = extractor.execute(image_path, label_path, label=ul)
            # Convert arrays of size 1 to floats, but skip tuples and strings
            for key, item in result.items():
                if isinstance(item, np.ndarray):
                    if item.size == 1:
                        result[key] = float(item)
                # print(key, item)
            results[ul] = result
        except BaseException as e:
            extractor = fe.RadiomicsFeatureExtractor(geometryTolerance=10**3)
            print('error enciuntered:', source_vol + '_' + seg, sub, ses, ul, e)
            error_list.append((ul, e))
            results[ul] = {}
            continue

    # Store the extracted radiomic features
    pickle_out(results, file_name)
    os.remove(working_on_file)

    # Save Error txt file with list of error producing labels
    if len(error_list):
        with open(error_file_name.format(len(error_list)), 'w') as output:
            output.write(str(error_list))
        print('Radiomics stored to', file_name, len(error_list), 'errors encountered')
    else:
        print('Radiomics stored to', file_name)
    bar.finish()

# Filesystem parameters
PARALLEL = True
OVERWRITE = False
tools = ['samseg', 'freesurfer', 'fastsurfer', 'vuno']
source_vols = ['MEMPRAGE', 'BRAIN', 'DTI']

# We can use one general formatting rule to access Volumes, Segmenatations and store Radiomic Features
dir_format = get_root('data', '{}_{}', '{}', '{}.{}')

# PyRadiomics parameters
params = {'verbose': False, 'geometryTolerance': 10 ** -3}

# Select the volume type the radiomic features are calculated on
for source_vol in source_vols:
    # Finding all files using volume data
    regex = dir_format.format('vols', source_vol, '*', '*', 'nii.gz')
    # Split each file into a subject and session tuple
    files = [file.split(os.sep)[-2:] for file in glob(regex)]
    files = multiply_for_each_tool(files)
    files = check_if_done(files, remove_tmps=True)

    if __name__ == '__main__':
        if PARALLEL:
            with Pool(processes=cpu_count()) as pool:
                pool.map(compute_radiomics, files)
        else:
            for file in files:
                compute_radiomics(file)
