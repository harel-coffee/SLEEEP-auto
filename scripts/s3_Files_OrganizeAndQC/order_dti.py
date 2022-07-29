import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from glob import glob
import os
from utils import get_root, sub_lut
from shutil import copy
files = glob(get_root('data', 'Oslo_SLEEEP_DTI', '*.nii.gz'))

#lut = {'600': '001',
#       '601': '002',
#       '602': '003',
#       '603': '004',
#       '604': '005',
#       '606': '006',
#       '607': '007',
#       '608': '008',
#       '612': '009',
#       '614': '010',
#       '616': '011',
#       '617': '012',
#       '618': '013',
#       '619': '014',
#       '621': '015',
#       '623': '016',
#       '624': '017',
#       '625': '018',
#       '626': '019',
#       '627': '020',
#       '628': '021',
#       '629': '022',
#       '631': '023',
#       '632': '024',
#       '633': '025',
#       '634': '026',
#       '635': '027',
#       '636': '028',
#       '637': '029',
#       '638': '030',
#      '640': '032',
#       '641': '033',
#       '642': '034',
#       '643': '035',
#       '644': '036',
#       '645': '037',
#       '646': '038',
#       '647': '039',
#       '648': '040',
#       '650': '041',
#       '651': '042',
#       '653': '043',
#       '654': '044',
#       '655': '045',
#       '656': '046',
#       '657': '047',
#       '658': '048'}

error_list = []
for file in files:
    # identify timepoint and subject id
    time_point, old_id = os.path.basename(file).split(os.extsep)[0].split('_')[-2:]
    new_id = lut[old_id]
    
    #  we take the file name from the similar files in the MEMPRAGE data
    cousins = glob(get_root('data', 'vols_MEMPRAGE', new_id, '*.nii.gz'))
    cousins.sort()
    try:
        fname = os.path.basename(cousins[int(time_point[-1])-1])
    except:
        error_list.append(file)
        fname = '20990919_195959.nii.gz'
    # store file
    target_dir = get_root('data', 'dwis_DTI', new_id)
    os.makedirs(target_dir, exist_ok=True)
    copy(file, os.path.join(target_dir, fname))
    print('Created:', os.path.join(target_dir, fname)[len(get_root()):])

print('\n ERRORS occurred on the following files:')
for error in error_list:
    print(error[len(get_root()):])
print('Source folder probably contained more data than cousins to red from.')