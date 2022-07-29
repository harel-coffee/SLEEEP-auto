'''

    This utility file contains some pretty usefull lookup-tables. Mainly:
    - freeSurfer_lut(input_value, output_type: str)
      INPUT_VALUE: TAKES 'label' value as integer OR anatomical 'name' as string OR 'rgb' color code as len 3 list
        and turns it into a 'label', 'name' or 'rgb'. e.g:
        freeSurfer_lut(2, 'name') >> 'Left-Cerebral-White-Matter'

      OUTPUT_VALUE:
    - is_deprived: TAKES a subject id and returns if the subject is sleep deprived (True) or is a control (False)

    August 2021, Maarten Poiort (maartenpoirot@gmail.com)

'''

import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
import nibabel as nib
import pandas as pd
from glob import glob
import os
from collections import Iterable, Generator


def sanitize_storage_path(storage_path=None, desc='Unnamed', ext='') -> str:
		"""
			Takes input as string and sanitizes it for: 
			1) No dir, nor fname given: 	Use default fname
			2) Just dir, no fname given:	Combine dir with default fname
			3) Dir and fname given:			Check for fname extension
			Returns sanitized storage_path as string.
		"""

		# Setup a default file name
		tm_stamp = datetime.now().strftime('%Y%m%d-%H%M')
		default_fname = desc + tm_stamp + ext
		
		# Sanitize input:
		if not storage_path:
			# 1) If no storage location was provided: 
			#    Store at location using default filename
			return default_fname
		elif os.path.isdir(storage_path):
			# 2) If only a storace directory was provded:
			#    Store at this location using default filename
			return os.path.join(storage_path, default_fname)
		else:
			# 3) If everything was provided:
			#    Check file name extension and save this location
			if not storage_path[len(ext):] == ext:
				storage_path += ext
			return storage_path


class Logger:
	"""
		Logger works like the Python 3+ print function, but 
		it also writes to a file and has handy logging modules like:
		- timestr: returns the current time as formatted string
		- collect debug information
	"""
	def __init__(self, storage_path=None):

		self.storage_path = sanitize_storage_path(storage_path, desc='Logger', ext='.txt')
		with open(self.storage_path, 'a') as log_file:
			print(timestr(), 'log stored at', self.storage_path)
			log_file.write(timestr() + ' log stored at ' + self.storage_path + '\n')

		self.debug = {}


	def __call__(self, *args):
		# First, print the log entry
		print(timestr(), *args)
		# Then write the log entry to file
		with open(self.storage_path, 'a') as log_file:
			log_file.write(' '.join([timestr()] + [str(arg) for arg in list(args)]) + '\n')


def get_root(*args) -> str:
    if os.name == 'posix':
        return os.path.join(r'/data/projects/depredict/repositories/SLEEEP/', *args)
    else:
        return os.path.join(r'D:\repositories\SLEEEP', *args)

def fisher_mean(rho_list):
    rho_list = [r for r in rho_list if not r == np.nan]
    rho_list = [0.9999999999999 if r >= 1 else r for r in rho_list]
    rho_list = [0.0000000000001 if r <= 0 else r for r in rho_list]
    # fisher transform            np.log((1 + r)/(1 - r))
    # inverse fisher transform:   (np.exp(2 * z) - 1)/(np.exp(2 * z) + 1)
    return np.tanh(np.nanmean(np.arctanh(rho_list)))

def pickle_in(source: str) -> dict:
    with open(source, 'rb') as file:
        return pickle.load(file)


def pickle_out(obj: dict, target: str):
    with open(target, 'wb') as file:
        pickle.dump(obj, file)


# header_type = list[str]
class Headers(list):
    def __init__(self, hdrs=[]):
        self.extend(hdrs)

    def get_idx(self, field_name: str) -> int:
        return next(idx for idx, content in enumerate(self) if content == field_name)


def log(logfile, *args):
    with open(logfile, 'a') as lf:
        print(*args)
        lf.write(' '.join(args) + '\n')


# TIMESTR: return current time as string
def timestr() -> str: return datetime.now().strftime('%H:%M')


# RM_DUMPLICATES: takes Iterable and returns the same list without duplicates
def rm_duplicates(str_in: Iterable) -> list: return list(dict.fromkeys(str_in))


# SLICE_FNAME: full path -> sliced file
def slice_fname(str_in: str, *slc: slice) -> str: return str_in.split(os.sep)[-1][slice(*slc)]


def _flush_nifti(file_path: str):
    nib.save(nib.Nifti1Image(np.zeros(nib.load(file_path).get_fdata().shape), np.eye(4)), file_path)


def _compress_nifti(file_path: str):
    nib.save(nib.load(file_path), file_path + '.gz')
    os.remove(file_path)


def _decompress_nifti(file_path: str):
    nib.save(nib.load(file_path), file_path[:-3])
    os.remove(file_path)


def _txt2csv(file_path: str):
    new_file_path = file_path.split(os.extsep)[0] + '.csv'
    pd.read_csv(file_path, sep='\t', encoding='utf-8').to_csv(new_file_path)


def _recursive_op(operation, root, filetype='*.nii.gz'):
    if operation == 'PERMANENTLY DELETE':
        method, message = _flush_nifti, 'flushed'
    elif operation == 'compress':
        method, message = _compress_nifti, 'compressed'
    elif operation == 'decompress':
        method, message = _compress_nifti, 'decompressed'
    elif operation == 'convert':
        method, message = _txt2csv, 'converted'
    else:
        raise ValueError('No proper operation designated')

    for i in range(10):
        regex = root + '*'.join(os.sep * i) + filetype
        files = glob(regex)
        if not len(files):
            continue
        go_through = input('Found ' + str(len(files)) + ' .nii.gz files in "' + regex + '". \
		Are you sure you want to ' + operation + ' their content? y/[n]')
        if go_through == 'y':
            for file in tqdm(files):
                method(file)
            print('Files' + message + '.')
        else:
            print('Response was "' + str(go_through) + '". Files were not' + message + '.')


def recursive_nifti_flush(root: str):
    _recursive_op('PERMANENTLY FLUSH', root, filetype='*.nii.gz')


def recursive_nifti_compress(root: str):
    _recursive_op('compress', root, filetype='*.nii.gz')


def recursive_nifti_decompress(root: str):
    _recursive_op('decompress', root, filetype='*.nii.gz')


def recursive_csv2txt(root: str):
    _recursive_op('convert', root, filetype='*.txt')


def flatten(container: Iterable) -> Generator:
    """
	:param container: Iterable to be flattened
	:return: flat Generator
	"""
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def freeSurfer_lut(input_value, output_type: str):
    lut = {'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                     27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                     52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                     77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                     109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                     129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149,
                     150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 180,
                     181, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
                     212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 250, 251, 252, 253, 254, 255,
                     256, 258, 259, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
                     350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409,
                     410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
                     430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 500, 501, 502, 503, 504, 505, 506, 507, 508, 550,
                     551, 552, 553, 554, 555, 556, 557, 558, 600, 701, 702, 703, 999, 1000, 1001, 1002, 1003, 1004,
                     1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
                     1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 2000,
                     2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
                     2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
                     2033, 2034, 2035, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012,
                     3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028,
                     3029, 3030, 3031, 3032, 3033, 3034, 3035, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008,
                     4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024,
                     4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 1100, 1101, 1102, 1103, 1104,
                     1200, 1201, 1202, 1205, 1206, 1207, 1210, 1211, 1212, 1105, 1106, 1107, 1108, 1109, 1110, 1111,
                     1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127,
                     1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143,
                     1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159,
                     1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175,
                     1176, 1177, 1178, 1179, 1180, 1181, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109,
                     2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125,
                     2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141,
                     2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157,
                     2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173,
                     2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2200, 2201, 2202, 2205, 2206, 2207, 2210, 2211,
                     2212],
           'name': ['Unknown', 'Left-Cerebral-Exterior', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex',
                    'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-Exterior',
                    'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex', 'Left-Thalamus', 'Left-Thalamus-Proper',
                    'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem',
                    'Left-Hippocampus', 'Left-Amygdala', 'Left-Insula', 'Left-Operculum', 'Line-1', 'Line-2', 'Line-3',
                    'CSF', 'Left-Lesion', 'Left-Accumbens-area', 'Left-Substancia-Nigra', 'Left-VentralDC',
                    'Left-undetermined', 'Left-vessel', 'Left-choroid-plexus', 'Left-F3orb', 'Left-lOg', 'Left-aOg',
                    'Left-mOg', 'Left-pOg', 'Left-Stellate', 'Left-Porg', 'Left-Aorg', 'Right-Cerebral-Exterior',
                    'Right-Cerebral-White-Matter', 'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle',
                    'Right-Inf-Lat-Vent', 'Right-Cerebellum-Exterior', 'Right-Cerebellum-White-Matter',
                    'Right-Cerebellum-Cortex', 'Right-Thalamus', 'Right-Thalamus-Proper', 'Right-Caudate',
                    'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Insula',
                    'Right-Operculum', 'Right-Lesion', 'Right-Accumbens-area', 'Right-Substancia-Nigra',
                    'Right-VentralDC', 'Right-undetermined', 'Right-vessel', 'Right-choroid-plexus', 'Right-F3orb',
                    'Right-lOg', 'Right-aOg', 'Right-mOg', 'Right-pOg', 'Right-Stellate', 'Right-Porg', 'Right-Aorg',
                    '5th-Ventricle', 'Left-Interior', 'Right-Interior', 'Left-Lateral-Ventricles',
                    'Right-Lateral-Ventricles', 'WM-hypointensities', 'Left-WM-hypointensities',
                    'Right-WM-hypointensities', 'non-WM-hypointensities', 'Left-non-WM-hypointensities',
                    'Right-non-WM-hypointensities', 'Left-F1', 'Right-F1', 'Optic-Chiasm', 'Corpus_Callosum',
                    'Left-Amygdala-Anterior', 'Right-Amygdala-Anterior', 'Dura', 'Left-wm-intensity-abnormality',
                    'Left-caudate-intensity-abnormality', 'Left-putamen-intensity-abnormality',
                    'Left-accumbens-intensity-abnormality', 'Left-pallidum-intensity-abnormality',
                    'Left-amygdala-intensity-abnormality', 'Left-hippocampus-intensity-abnormality',
                    'Left-thalamus-intensity-abnormality', 'Left-VDC-intensity-abnormality',
                    'Right-wm-intensity-abnormality', 'Right-caudate-intensity-abnormality',
                    'Right-putamen-intensity-abnormality', 'Right-accumbens-intensity-abnormality',
                    'Right-pallidum-intensity-abnormality', 'Right-amygdala-intensity-abnormality',
                    'Right-hippocampus-intensity-abnormality', 'Right-thalamus-intensity-abnormality',
                    'Right-VDC-intensity-abnormality', 'Epidermis', 'Conn-Tissue', 'SC-Fat/Muscle', 'Cranium', 'CSF-SA',
                    'Muscle', 'Ear', 'Adipose', 'Spinal-Cord', 'Soft-Tissue', 'Nerve', 'Bone', 'Air', 'Orbital-Fat',
                    'Tongue', 'Nasal-Structures', 'Globe', 'Teeth', 'Left-Caudate/Putamen', 'Right-Caudate/Putamen',
                    'Left-Claustrum', 'Right-Claustrum', 'Cornea', 'Diploe', 'Vitreous-Humor', 'Lens', 'Aqueous-Humor',
                    'Outer-Table', 'Inner-Table', 'Periosteum', 'Endosteum', 'R/C/S', 'Iris', 'SC-Adipose/Muscle',
                    'SC-Tissue', 'Orbital-Adipose', 'Left-IntCapsule-Ant', 'Right-IntCapsule-Ant',
                    'Left-IntCapsule-Pos', 'Right-IntCapsule-Pos', 'Left-Cerebral-WM-unmyelinated',
                    'Right-Cerebral-WM-unmyelinated', 'Left-Cerebral-WM-myelinated', 'Right-Cerebral-WM-myelinated',
                    'Left-Subcortical-Gray-Matter', 'Right-Subcortical-Gray-Matter', 'Skull', 'Posterior-fossa',
                    'Scalp', 'Hematoma', 'Left-Cortical-Dysplasia', 'Right-Cortical-Dysplasia',
                    'Corpus_Callosum', 'Left-hippocampal_fissure', 'Left-CADG-head', 'Left-subiculum', 'Left-fimbria',
                    'Right-hippocampal_fissure', 'Right-CADG-head', 'Right-subiculum', 'Right-fimbria', 'alveus',
                    'perforant_pathway', 'parasubiculum', 'presubiculum', 'subiculum', 'CA1', 'CA2', 'CA3', 'CA4',
                    'GC-DG', 'HATA', 'fimbria', 'lateral_ventricle', 'molecular_layer_HP', 'hippocampal_fissure',
                    'entorhinal_cortex', 'molecular_layer_subiculum', 'Amygdala', 'Cerebral_White_Matter',
                    'Cerebral_Cortex', 'Inf_Lat_Vent', 'Perirhinal', 'Cerebral_White_Matter_Edge', 'Background',
                    'Ectorhinal', 'Fornix', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior',
                    'CC_Anterior', 'Voxel-Unchanged', 'CSF-ExtraCerebral', 'Head-ExtraCerebral', 'Aorta', 'Left-Common-IliacA', 'Right-Common-IliacA',
                    'Left-External-IliacA', 'Right-External-IliacA', 'Left-Internal-IliacA', 'Right-Internal-IliacA',
                    'Left-Lateral-SacralA', 'Right-Lateral-SacralA', 'Left-ObturatorA', 'Right-ObturatorA',
                    'Left-Internal-PudendalA', 'Right-Internal-PudendalA', 'Left-UmbilicalA', 'Right-UmbilicalA',
                    'Left-Inf-RectalA', 'Right-Inf-RectalA', 'Left-Common-IliacV', 'Right-Common-IliacV',
                    'Left-External-IliacV', 'Right-External-IliacV', 'Left-Internal-IliacV', 'Right-Internal-IliacV',
                    'Left-ObturatorV', 'Right-ObturatorV', 'Left-Internal-PudendalV', 'Right-Internal-PudendalV',
                    'Pos-Lymph', 'Neg-Lymph', 'V1', 'V2', 'BA44', 'BA45', 'BA4a', 'BA4p', 'BA6', 'BA2', 'BA1_old',
                    'BAun2', 'BA1', 'BA2b', 'BA3a', 'BA3b', 'MT', 'AIPS_AIP_l', 'AIPS_AIP_r', 'AIPS_VIP_l',
                    'AIPS_VIP_r', 'IPL_PFcm_l', 'IPL_PFcm_r', 'IPL_PF_l', 'IPL_PFm_l', 'IPL_PFm_r', 'IPL_PFop_l',
                    'IPL_PFop_r', 'IPL_PF_r', 'IPL_PFt_l', 'IPL_PFt_r', 'IPL_PGa_l', 'IPL_PGa_r', 'IPL_PGp_l',
                    'IPL_PGp_r', 'Visual_V3d_l', 'Visual_V3d_r', 'Visual_V4_l', 'Visual_V4_r', 'Visual_V5_b',
                    'Visual_VP_l', 'Visual_VP_r', 'right_CA2/3', 'right_alveus', 'right_CA1', 'right_fimbria',
                    'right_presubiculum', 'right_hippocampal_fissure', 'right_CA4/DG', 'right_subiculum',
                    'right_fornix', 'left_CA2/3', 'left_alveus', 'left_CA1', 'left_fimbria', 'left_presubiculum',
                    'left_hippocampal_fissure', 'left_CA4/DG', 'left_subiculum', 'left_fornix', 'Tumor', 'CSF-FSL-FAST',
                    'GrayMatter-FSL-FAST', 'WhiteMatter-FSL-FAST', 'SUSPICIOUS', 'ctx-lh-unknown', 'ctx-lh-bankssts',
                    'ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-corpuscallosum',
                    'ctx-lh-cuneus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform', 'ctx-lh-inferiorparietal',
                    'ctx-lh-inferiortemporal', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                    'ctx-lh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-lh-medialorbitofrontal',
                    'ctx-lh-middletemporal', 'ctx-lh-parahippocampal', 'ctx-lh-paracentral', 'ctx-lh-parsopercularis',
                    'ctx-lh-parsorbitalis', 'ctx-lh-parstriangularis', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral',
                    'ctx-lh-posteriorcingulate', 'ctx-lh-precentral', 'ctx-lh-precuneus',
                    'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal', 'ctx-lh-superiorfrontal',
                    'ctx-lh-superiorparietal', 'ctx-lh-superiortemporal', 'ctx-lh-supramarginal', 'ctx-lh-frontalpole',
                    'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 'ctx-lh-insula', 'ctx-rh-unknown',
                    'ctx-rh-bankssts', 'ctx-rh-caudalanteriorcingulate', 'ctx-rh-caudalmiddlefrontal',
                    'ctx-rh-corpuscallosum', 'ctx-rh-cuneus', 'ctx-rh-entorhinal', 'ctx-rh-fusiform',
                    'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal', 'ctx-rh-isthmuscingulate',
                    'ctx-rh-lateraloccipital', 'ctx-rh-lateralorbitofrontal', 'ctx-rh-lingual',
                    'ctx-rh-medialorbitofrontal', 'ctx-rh-middletemporal', 'ctx-rh-parahippocampal',
                    'ctx-rh-paracentral', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-parstriangularis',
                    'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate', 'ctx-rh-precentral',
                    'ctx-rh-precuneus', 'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal',
                    'ctx-rh-superiorfrontal', 'ctx-rh-superiorparietal', 'ctx-rh-superiortemporal',
                    'ctx-rh-supramarginal', 'ctx-rh-frontalpole', 'ctx-rh-temporalpole', 'ctx-rh-transversetemporal',
                    'ctx-rh-insula', 'wm-lh-unknown', 'wm-lh-bankssts', 'wm-lh-caudalanteriorcingulate',
                    'wm-lh-caudalmiddlefrontal', 'wm-lh-corpuscallosum', 'wm-lh-cuneus', 'wm-lh-entorhinal',
                    'wm-lh-fusiform', 'wm-lh-inferiorparietal', 'wm-lh-inferiortemporal', 'wm-lh-isthmuscingulate',
                    'wm-lh-lateraloccipital', 'wm-lh-lateralorbitofrontal', 'wm-lh-lingual',
                    'wm-lh-medialorbitofrontal', 'wm-lh-middletemporal', 'wm-lh-parahippocampal', 'wm-lh-paracentral',
                    'wm-lh-parsopercularis', 'wm-lh-parsorbitalis', 'wm-lh-parstriangularis', 'wm-lh-pericalcarine',
                    'wm-lh-postcentral', 'wm-lh-posteriorcingulate', 'wm-lh-precentral', 'wm-lh-precuneus',
                    'wm-lh-rostralanteriorcingulate', 'wm-lh-rostralmiddlefrontal', 'wm-lh-superiorfrontal',
                    'wm-lh-superiorparietal', 'wm-lh-superiortemporal', 'wm-lh-supramarginal', 'wm-lh-frontalpole',
                    'wm-lh-temporalpole', 'wm-lh-transversetemporal', 'wm-lh-insula', 'wm-rh-unknown', 'wm-rh-bankssts',
                    'wm-rh-caudalanteriorcingulate', 'wm-rh-caudalmiddlefrontal', 'wm-rh-corpuscallosum',
                    'wm-rh-cuneus', 'wm-rh-entorhinal', 'wm-rh-fusiform', 'wm-rh-inferiorparietal',
                    'wm-rh-inferiortemporal', 'wm-rh-isthmuscingulate', 'wm-rh-lateraloccipital',
                    'wm-rh-lateralorbitofrontal', 'wm-rh-lingual', 'wm-rh-medialorbitofrontal', 'wm-rh-middletemporal',
                    'wm-rh-parahippocampal', 'wm-rh-paracentral', 'wm-rh-parsopercularis', 'wm-rh-parsorbitalis',
                    'wm-rh-parstriangularis', 'wm-rh-pericalcarine', 'wm-rh-postcentral', 'wm-rh-posteriorcingulate',
                    'wm-rh-precentral', 'wm-rh-precuneus', 'wm-rh-rostralanteriorcingulate',
                    'wm-rh-rostralmiddlefrontal', 'wm-rh-superiorfrontal', 'wm-rh-superiorparietal',
                    'wm-rh-superiortemporal', 'wm-rh-supramarginal', 'wm-rh-frontalpole', 'wm-rh-temporalpole',
                    'wm-rh-transversetemporal', 'wm-rh-insula', 'ctx-lh-Unknown', 'ctx-lh-Corpus_callosum',
                    'ctx-lh-G_and_S_Insula_ONLY_AVERAGE', 'ctx-lh-G_cingulate-Isthmus', 'ctx-lh-G_cingulate-Main_part',
                    'ctx-lh-G_cingulate-caudal_ACC', 'ctx-lh-G_cingulate-rostral_ACC', 'ctx-lh-G_cingulate-posterior',
                    'ctx-lh-S_cingulate-caudal_ACC', 'ctx-lh-S_cingulate-rostral_ACC', 'ctx-lh-S_cingulate-posterior',
                    'ctx-lh-S_pericallosal-caudal', 'ctx-lh-S_pericallosal-rostral', 'ctx-lh-S_pericallosal-posterior',
                    'ctx-lh-G_cuneus', 'ctx-lh-G_frontal_inf-Opercular_part', 'ctx-lh-G_frontal_inf-Orbital_part',
                    'ctx-lh-G_frontal_inf-Triangular_part', 'ctx-lh-G_frontal_middle', 'ctx-lh-G_frontal_superior',
                    'ctx-lh-G_frontomarginal', 'ctx-lh-G_insular_long', 'ctx-lh-G_insular_short',
                    'ctx-lh-G_and_S_occipital_inferior', 'ctx-lh-G_occipital_middle', 'ctx-lh-G_occipital_superior',
                    'ctx-lh-G_occipit-temp_lat-Or_fusiform', 'ctx-lh-G_occipit-temp_med-Lingual_part',
                    'ctx-lh-G_occipit-temp_med-Parahippocampal_part', 'ctx-lh-G_orbital', 'ctx-lh-G_paracentral',
                    'ctx-lh-G_parietal_inferior-Angular_part', 'ctx-lh-G_parietal_inferior-Supramarginal_part',
                    'ctx-lh-G_parietal_superior', 'ctx-lh-G_postcentral', 'ctx-lh-G_precentral', 'ctx-lh-G_precuneus',
                    'ctx-lh-G_rectus', 'ctx-lh-G_subcallosal', 'ctx-lh-G_subcentral', 'ctx-lh-G_temporal_inferior',
                    'ctx-lh-G_temporal_middle', 'ctx-lh-G_temp_sup-G_temp_transv_and_interm_S',
                    'ctx-lh-G_temp_sup-Lateral_aspect', 'ctx-lh-G_temp_sup-Planum_polare',
                    'ctx-lh-G_temp_sup-Planum_tempolare', 'ctx-lh-G_and_S_transverse_frontopolar',
                    'ctx-lh-Lat_Fissure-ant_sgt-ramus_horizontal', 'ctx-lh-Lat_Fissure-ant_sgt-ramus_vertical',
                    'ctx-lh-Lat_Fissure-post_sgt', 'ctx-lh-Medial_wall', 'ctx-lh-Pole_occipital',
                    'ctx-lh-Pole_temporal', 'ctx-lh-S_calcarine', 'ctx-lh-S_central', 'ctx-lh-S_central_insula',
                    'ctx-lh-S_cingulate-Main_part_and_Intracingulate', 'ctx-lh-S_cingulate-Marginalis_part',
                    'ctx-lh-S_circular_insula_anterior', 'ctx-lh-S_circular_insula_inferior',
                    'ctx-lh-S_circular_insula_superior', 'ctx-lh-S_collateral_transverse_ant',
                    'ctx-lh-S_collateral_transverse_post', 'ctx-lh-S_frontal_inferior', 'ctx-lh-S_frontal_middle',
                    'ctx-lh-S_frontal_superior', 'ctx-lh-S_frontomarginal', 'ctx-lh-S_intermedius_primus-Jensen',
                    'ctx-lh-S_intraparietal-and_Parietal_transverse', 'ctx-lh-S_occipital_anterior',
                    'ctx-lh-S_occipital_middle_and_Lunatus', 'ctx-lh-S_occipital_superior_and_transversalis',
                    'ctx-lh-S_occipito-temporal_lateral', 'ctx-lh-S_occipito-temporal_medial_and_S_Lingual',
                    'ctx-lh-S_orbital-H_shapped', 'ctx-lh-S_orbital_lateral', 'ctx-lh-S_orbital_medial-Or_olfactory',
                    'ctx-lh-S_paracentral', 'ctx-lh-S_parieto_occipital', 'ctx-lh-S_pericallosal',
                    'ctx-lh-S_postcentral', 'ctx-lh-S_precentral-Inferior-part', 'ctx-lh-S_precentral-Superior-part',
                    'ctx-lh-S_subcentral_ant', 'ctx-lh-S_subcentral_post', 'ctx-lh-S_suborbital',
                    'ctx-lh-S_subparietal', 'ctx-lh-S_supracingulate', 'ctx-lh-S_temporal_inferior',
                    'ctx-lh-S_temporal_superior', 'ctx-lh-S_temporal_transverse', 'ctx-rh-Unknown',
                    'ctx-rh-Corpus_callosum', 'ctx-rh-G_and_S_Insula_ONLY_AVERAGE', 'ctx-rh-G_cingulate-Isthmus',
                    'ctx-rh-G_cingulate-Main_part', 'ctx-rh-G_cuneus', 'ctx-rh-G_frontal_inf-Opercular_part',
                    'ctx-rh-G_frontal_inf-Orbital_part', 'ctx-rh-G_frontal_inf-Triangular_part',
                    'ctx-rh-G_frontal_middle', 'ctx-rh-G_frontal_superior', 'ctx-rh-G_frontomarginal',
                    'ctx-rh-G_insular_long', 'ctx-rh-G_insular_short', 'ctx-rh-G_and_S_occipital_inferior',
                    'ctx-rh-G_occipital_middle', 'ctx-rh-G_occipital_superior', 'ctx-rh-G_occipit-temp_lat-Or_fusiform',
                    'ctx-rh-G_occipit-temp_med-Lingual_part', 'ctx-rh-G_occipit-temp_med-Parahippocampal_part',
                    'ctx-rh-G_orbital', 'ctx-rh-G_paracentral', 'ctx-rh-G_parietal_inferior-Angular_part',
                    'ctx-rh-G_parietal_inferior-Supramarginal_part', 'ctx-rh-G_parietal_superior',
                    'ctx-rh-G_postcentral', 'ctx-rh-G_precentral', 'ctx-rh-G_precuneus', 'ctx-rh-G_rectus',
                    'ctx-rh-G_subcallosal', 'ctx-rh-G_subcentral', 'ctx-rh-G_temporal_inferior',
                    'ctx-rh-G_temporal_middle', 'ctx-rh-G_temp_sup-G_temp_transv_and_interm_S',
                    'ctx-rh-G_temp_sup-Lateral_aspect', 'ctx-rh-G_temp_sup-Planum_polare',
                    'ctx-rh-G_temp_sup-Planum_tempolare', 'ctx-rh-G_and_S_transverse_frontopolar',
                    'ctx-rh-Lat_Fissure-ant_sgt-ramus_horizontal', 'ctx-rh-Lat_Fissure-ant_sgt-ramus_vertical',
                    'ctx-rh-Lat_Fissure-post_sgt', 'ctx-rh-Medial_wall', 'ctx-rh-Pole_occipital',
                    'ctx-rh-Pole_temporal', 'ctx-rh-S_calcarine', 'ctx-rh-S_central', 'ctx-rh-S_central_insula',
                    'ctx-rh-S_cingulate-Main_part_and_Intracingulate', 'ctx-rh-S_cingulate-Marginalis_part',
                    'ctx-rh-S_circular_insula_anterior', 'ctx-rh-S_circular_insula_inferior',
                    'ctx-rh-S_circular_insula_superior', 'ctx-rh-S_collateral_transverse_ant',
                    'ctx-rh-S_collateral_transverse_post', 'ctx-rh-S_frontal_inferior', 'ctx-rh-S_frontal_middle',
                    'ctx-rh-S_frontal_superior', 'ctx-rh-S_frontomarginal', 'ctx-rh-S_intermedius_primus-Jensen',
                    'ctx-rh-S_intraparietal-and_Parietal_transverse', 'ctx-rh-S_occipital_anterior',
                    'ctx-rh-S_occipital_middle_and_Lunatus', 'ctx-rh-S_occipital_superior_and_transversalis',
                    'ctx-rh-S_occipito-temporal_lateral', 'ctx-rh-S_occipito-temporal_medial_and_S_Lingual',
                    'ctx-rh-S_orbital-H_shapped', 'ctx-rh-S_orbital_lateral', 'ctx-rh-S_orbital_medial-Or_olfactory',
                    'ctx-rh-S_paracentral', 'ctx-rh-S_parieto_occipital', 'ctx-rh-S_pericallosal',
                    'ctx-rh-S_postcentral', 'ctx-rh-S_precentral-Inferior-part', 'ctx-rh-S_precentral-Superior-part',
                    'ctx-rh-S_subcentral_ant', 'ctx-rh-S_subcentral_post', 'ctx-rh-S_suborbital',
                    'ctx-rh-S_subparietal', 'ctx-rh-S_supracingulate', 'ctx-rh-S_temporal_inferior',
                    'ctx-rh-S_temporal_superior', 'ctx-rh-S_temporal_transverse', 'ctx-rh-G_cingulate-caudal_ACC',
                    'ctx-rh-G_cingulate-rostral_ACC', 'ctx-rh-G_cingulate-posterior', 'ctx-rh-S_cingulate-caudal_ACC',
                    'ctx-rh-S_cingulate-rostral_ACC', 'ctx-rh-S_cingulate-posterior', 'ctx-rh-S_pericallosal-caudal',
                    'ctx-rh-S_pericallosal-rostral', 'ctx-rh-S_pericallosal-posterior'],
           'rgb': [[0, 0, 0], [70, 130, 180], [245, 245, 245], [205, 62, 78], [120, 18, 134], [196, 58, 250],
                   [0, 148, 0], [220, 248, 164], [230, 148, 34], [0, 118, 14], [0, 118, 14], [122, 186, 220],
                   [236, 13, 176], [12, 48, 255], [204, 182, 142], [42, 204, 164], [119, 159, 176], [220, 216, 20],
                   [103, 255, 255], [80, 196, 98], [60, 58, 210], [60, 58, 210], [60, 58, 210], [60, 58, 210],
                   [60, 60, 60], [255, 165, 0], [255, 165, 0], [0, 255, 127], [165, 42, 42], [135, 206, 235],
                   [160, 32, 240], [0, 200, 200], [100, 50, 100], [135, 50, 74], [122, 135, 50], [51, 50, 135],
                   [74, 155, 60], [120, 62, 43], [74, 155, 60], [122, 135, 50], [70, 130, 180], [0, 225, 0],
                   [205, 62, 78], [120, 18, 134], [196, 58, 250], [0, 148, 0], [220, 248, 164], [230, 148, 34],
                   [0, 118, 14], [0, 118, 14], [122, 186, 220], [236, 13, 176], [13, 48, 255], [220, 216, 20],
                   [103, 255, 255], [80, 196, 98], [60, 58, 210], [255, 165, 0], [255, 165, 0], [0, 255, 127],
                   [165, 42, 42], [135, 206, 235], [160, 32, 240], [0, 200, 221], [100, 50, 100], [135, 50, 74],
                   [122, 135, 50], [51, 50, 135], [74, 155, 60], [120, 62, 43], [74, 155, 60], [122, 135, 50],
                   [120, 190, 150], [122, 135, 50], [122, 135, 50], [120, 18, 134], [120, 18, 134], [200, 70, 255],
                   [255, 148, 10], [255, 148, 10], [164, 108, 226], [164, 108, 226], [164, 108, 226], [255, 218, 185],
                   [255, 218, 185], [234, 169, 30], [250, 255, 50], [205, 10, 125], [205, 10, 125], [160, 32, 240],
                   [124, 140, 178], [125, 140, 178], [126, 140, 178], [127, 140, 178], [124, 141, 178], [124, 142, 178],
                   [124, 143, 178], [124, 144, 178], [124, 140, 179], [124, 140, 178], [125, 140, 178], [126, 140, 178],
                   [127, 140, 178], [124, 141, 178], [124, 142, 178], [124, 143, 178], [124, 144, 178], [124, 140, 179],
                   [255, 20, 147], [205, 179, 139], [238, 238, 209], [200, 200, 200], [74, 255, 74], [238, 0, 0],
                   [0, 0, 139], [173, 255, 47], [133, 203, 229], [26, 237, 57], [34, 139, 34], [30, 144, 255],
                   [147, 19, 173], [238, 59, 59], [221, 39, 200], [238, 174, 238], [255, 0, 0], [72, 61, 139],
                   [21, 39, 132], [21, 39, 132], [65, 135, 20], [65, 135, 20], [134, 4, 160], [221, 226, 68],
                   [255, 255, 254], [52, 209, 226], [239, 160, 223], [70, 130, 180], [70, 130, 181], [139, 121, 94],
                   [224, 224, 224], [255, 0, 0], [205, 205, 0], [238, 238, 209], [139, 121, 94], [238, 59, 59],
                   [238, 59, 59], [238, 59, 59], [62, 10, 205], [62, 10, 205], [0, 118, 14], [0, 118, 14],
                   [220, 216, 21], [220, 216, 21], [122, 186, 220], [122, 186, 220], [255, 165, 0], [14, 48, 255],
                   [166, 42, 42], [121, 18, 134], [73, 61, 139], [73, 62, 139], [250, 255, 50], [0, 196, 255], [255, 164, 164],
                   [196, 196, 0], [0, 100, 255], [128, 196, 164], [0, 126, 75], [128, 96, 64], [0, 50, 128],
                   [255, 204, 153], [255, 128, 128], [255, 255, 0], [64, 0, 64], [0, 0, 255], [255, 0, 0],
                   [128, 128, 255], [0, 128, 0], [196, 160, 128], [32, 200, 255], [128, 255, 128], [204, 153, 204],
                   [121, 17, 136], [128, 0, 0], [128, 32, 255], [255, 204, 102], [128, 128, 128], [104, 255, 255],
                   [0, 226, 0], [205, 63, 78], [197, 58, 250], [33, 150, 250], [226, 0, 0], [100, 100, 100],
                   [197, 150, 250], [255, 0, 0], [0, 0, 64], [0, 0, 112], [0, 0, 160], [0, 0, 208], [0, 0, 255],
                   [0, 0, 0], [60,  60,  60], [150, 150, 200], [255, 0, 0], [255, 80, 0], [255, 160, 0], [255, 255, 0], [0, 255, 0], [255, 0, 160],
                   [255, 0, 255], [255, 50, 80], [80, 255, 50], [160, 255, 50], [160, 200, 255], [0, 255, 160],
                   [0, 0, 255], [80, 50, 255], [160, 0, 255], [255, 210, 0], [0, 160, 255], [255, 200, 80],
                   [255, 200, 160], [255, 80, 200], [255, 160, 200], [30, 255, 80], [80, 200, 255], [80, 255, 200],
                   [195, 255, 200], [120, 200, 20], [170, 10, 200], [20, 130, 180], [20, 180, 130], [206, 62, 78],
                   [121, 18, 134], [199, 58, 250], [1, 148, 0], [221, 248, 164], [231, 148, 34], [1, 118, 14],
                   [120, 118, 14], [123, 186, 221], [238, 13, 177], [123, 186, 220], [138, 13, 206], [238, 130, 176],
                   [218, 230, 76], [38, 213, 176], [1, 225, 176], [1, 225, 176], [200, 2, 100], [200, 2, 100],
                   [5, 200, 90], [5, 200, 90], [100, 5, 200], [25, 255, 100], [25, 255, 100], [230, 7, 100],
                   [230, 7, 100], [100, 5, 200], [150, 10, 200], [150, 10, 200], [175, 10, 176], [175, 10, 176],
                   [10, 100, 255], [10, 100, 255], [150, 45, 70], [150, 45, 70], [45, 200, 15], [45, 200, 15],
                   [227, 45, 100], [227, 45, 100], [227, 45, 100], [17, 85, 136], [119, 187, 102], [204, 68, 34],
                   [204, 0, 255], [221, 187, 17], [153, 221, 238], [51, 17, 17], [0, 119, 85], [20, 100, 200],
                   [17, 85, 137], [119, 187, 103], [204, 68, 35], [204, 0, 254], [221, 187, 16], [153, 221, 239],
                   [51, 17, 18], [0, 119, 86], [20, 100, 201], [254, 254, 254], [120, 18, 134], [205, 62, 78],
                   [0, 225, 0], [255, 100, 100], [25, 5, 25], [25, 100, 40], [125, 100, 160], [100, 25, 0],
                   [120, 70, 50], [220, 20, 100], [220, 20, 10], [180, 220, 140], [220, 60, 220], [180, 40, 120],
                   [140, 20, 140], [20, 30, 140], [35, 75, 50], [225, 140, 140], [200, 35, 75], [160, 100, 50],
                   [20, 220, 60], [60, 220, 60], [220, 180, 140], [20, 100, 50], [220, 60, 20], [120, 100, 60],
                   [220, 20, 20], [220, 180, 220], [60, 20, 220], [160, 140, 180], [80, 20, 140], [75, 50, 125],
                   [20, 220, 160], [20, 180, 140], [140, 220, 220], [80, 160, 20], [100, 0, 100], [70, 70, 70],
                   [150, 150, 200], [255, 192, 32], [25, 5, 25], [25, 100, 40], [125, 100, 160], [100, 25, 0],
                   [120, 70, 50], [220, 20, 100], [220, 20, 10], [180, 220, 140], [220, 60, 220], [180, 40, 120],
                   [140, 20, 140], [20, 30, 140], [35, 75, 50], [225, 140, 140], [200, 35, 75], [160, 100, 50],
                   [20, 220, 60], [60, 220, 60], [220, 180, 140], [20, 100, 50], [220, 60, 20], [120, 100, 60],
                   [220, 20, 20], [220, 180, 220], [60, 20, 220], [160, 140, 180], [80, 20, 140], [75, 50, 125],
                   [20, 220, 160], [20, 180, 140], [140, 220, 220], [80, 160, 20], [100, 0, 100], [70, 70, 70],
                   [150, 150, 200], [255, 192, 32], [230, 250, 230], [230, 155, 215], [130, 155, 95], [155, 230, 255],
                   [135, 185, 205], [35, 235, 155], [35, 235, 245], [75, 35, 115], [35, 195, 35], [75, 215, 135],
                   [115, 235, 115], [235, 225, 115], [220, 180, 205], [30, 115, 115], [55, 220, 180], [95, 155, 205],
                   [235, 35, 195], [195, 35, 195], [35, 75, 115], [235, 155, 205], [35, 195, 235], [135, 155, 195],
                   [35, 235, 235], [35, 75, 35], [195, 235, 35], [95, 115, 75], [175, 235, 115], [180, 205, 130],
                   [235, 35, 95], [235, 75, 115], [115, 35, 35], [175, 95, 235], [155, 255, 155], [185, 185, 185],
                   [105, 105, 55], [254, 191, 31], [230, 250, 230], [230, 155, 215], [130, 155, 95], [155, 230, 255],
                   [135, 185, 205], [35, 235, 155], [35, 235, 245], [75, 35, 115], [35, 195, 35], [75, 215, 135],
                   [115, 235, 115], [235, 225, 115], [220, 180, 205], [30, 115, 115], [55, 220, 180], [95, 155, 205],
                   [235, 35, 195], [195, 35, 195], [35, 75, 115], [235, 155, 205], [35, 195, 235], [135, 155, 195],
                   [35, 235, 235], [35, 75, 35], [195, 235, 35], [95, 115, 75], [175, 235, 115], [180, 205, 130],
                   [235, 35, 95], [235, 75, 115], [115, 35, 35], [175, 95, 235], [155, 255, 155], [185, 185, 185],
                   [105, 105, 55], [254, 191, 31], [0, 0, 0], [50, 50, 50], [180, 20, 30], [60, 25, 25], [25, 60, 60],
                   [25, 60, 61], [25, 90, 60], [25, 120, 60], [25, 150, 60], [25, 180, 60], [25, 210, 60],
                   [25, 150, 90], [25, 180, 90], [25, 210, 90], [180, 20, 20], [220, 20, 100], [140, 60, 60],
                   [180, 220, 140], [140, 100, 180], [180, 20, 140], [140, 20, 140], [21, 10, 10], [225, 140, 140],
                   [23, 60, 180], [180, 60, 180], [20, 220, 60], [60, 20, 140], [220, 180, 140], [65, 100, 20],
                   [220, 60, 20], [60, 100, 60], [20, 60, 220], [100, 100, 60], [220, 180, 220], [20, 180, 140],
                   [60, 140, 180], [25, 20, 140], [20, 60, 100], [60, 220, 20], [60, 20, 220], [220, 220, 100],
                   [180, 60, 60], [60, 60, 220], [220, 60, 220], [65, 220, 60], [25, 140, 20], [13, 0, 250],
                   [61, 20, 220], [61, 20, 60], [61, 60, 100], [25, 25, 25], [140, 20, 60], [220, 180, 20],
                   [63, 180, 180], [221, 20, 10], [21, 220, 20], [183, 100, 20], [221, 20, 100], [221, 60, 140],
                   [221, 20, 220], [61, 220, 220], [100, 200, 200], [10, 200, 200], [221, 220, 20], [141, 20, 100],
                   [61, 220, 100], [21, 220, 60], [141, 60, 20], [143, 20, 220], [61, 20, 180], [101, 60, 220],
                   [21, 20, 140], [221, 140, 20], [141, 100, 220], [101, 20, 20], [221, 100, 20], [181, 200, 20],
                   [21, 180, 140], [101, 100, 180], [181, 220, 20], [21, 140, 200], [21, 20, 240], [21, 20, 200],
                   [61, 180, 60], [61, 180, 250], [21, 20, 60], [101, 60, 60], [21, 220, 220], [21, 180, 180],
                   [223, 220, 60], [221, 60, 60], [0, 0, 0], [50, 50, 50], [180, 20, 30], [60, 25, 25], [25, 60, 60],
                   [180, 20, 20], [220, 20, 100], [140, 60, 60], [180, 220, 140], [140, 100, 180], [180, 20, 140],
                   [140, 20, 140], [21, 10, 10], [225, 140, 140], [23, 60, 180], [180, 60, 180], [20, 220, 60],
                   [60, 20, 140], [220, 180, 140], [65, 100, 20], [220, 60, 20], [60, 100, 60], [20, 60, 220],
                   [100, 100, 60], [220, 180, 220], [20, 180, 140], [60, 140, 180], [25, 20, 140], [20, 60, 100],
                   [60, 220, 20], [60, 20, 220], [220, 220, 100], [180, 60, 60], [60, 60, 220], [220, 60, 220],
                   [65, 220, 60], [25, 140, 20], [13, 0, 250], [61, 20, 220], [61, 20, 60], [61, 60, 100], [25, 25, 25],
                   [140, 20, 60], [220, 180, 20], [63, 180, 180], [221, 20, 10], [21, 220, 20], [183, 100, 20],
                   [221, 20, 100], [221, 60, 140], [221, 20, 220], [61, 220, 220], [100, 200, 200], [10, 200, 200],
                   [221, 220, 20], [141, 20, 100], [61, 220, 100], [21, 220, 60], [141, 60, 20], [143, 20, 220],
                   [61, 20, 180], [101, 60, 220], [21, 20, 140], [221, 140, 20], [141, 100, 220], [101, 20, 20],
                   [221, 100, 20], [181, 200, 20], [21, 180, 140], [101, 100, 180], [181, 220, 20], [21, 140, 200],
                   [21, 20, 240], [21, 20, 200], [61, 180, 60], [61, 180, 250], [21, 20, 60], [101, 60, 60],
                   [21, 220, 220], [21, 180, 180], [223, 220, 60], [221, 60, 60], [25, 60, 61], [25, 90, 60],
                   [25, 120, 60], [25, 150, 60], [25, 180, 60], [25, 210, 60], [25, 150, 90], [25, 180, 90],
                   [25, 210, 90]]}

    if output_type not in ['label', 'name', 'rgb']:
        raise ValueError('"output" is expected to be "label", "name" or "rgb"')

    if isinstance(input_value, int):
        return lut[output_type][lut['label'].index(input_value)]
    elif isinstance(input_value, str):
        return lut[output_type][lut['name'].index(input_value)]
    elif isinstance(input_value, list):
        return lut[output_type][lut['rgb'].index(input_value)]
    else:
        raise ValueError('"input" was not recognized as type int (label), str (name) or list (rgb).')


sub_lut = {'600':	'001',
       '601':	'002',
       '602':	'003',
       '603':	'004',
       '604':	'005',
       '606':	'006',
       '607':	'007',
       '608':	'008',
       '609':	'009',
       '612':	'010',
       '614':	'011',
       '616':	'012',
       '617':	'013',
       '618':	'014',
       '619':	'015',
       '621':	'016',
       '623':	'017',
       '624':	'018',
       '625':	'019',
       '626':	'020',
       '627':	'021',
       '628':	'022',
       '629':	'023',
       '631':	'024',
       '632':	'025',
       '633':	'026',
       '634':	'027',
       '635':	'028',
       '636':	'029',
       '637':	'030',
       '638':	'031',
       '639':	'032',
       '640':	'033',
       '641':	'034',
       '642':	'035',
       '643':	'036',
       '644':	'037',
       '645':	'038',
       '646':	'039',
       '647':	'040',
       '648':	'041',
       '650':	'042',
       '651':	'043',
       '653':	'044',
       '654':	'045',
       '655':	'046',
       '656':	'047',
       '657':	'048',
       '658':	'049'}


is_deprived = {'001': False,
            '002': True,
            '003': False,
            '004': True,
            '005': False,
            '006': True,
            '007': False,
            '008': True,
            '009': False,
            '010': False,
            '011': True,
            '012': False,
            '013': False,
            '014': False,
            '015': True,
            '016': True,
            '017': True,
            '018': True,
            '019': False,
            '020': True,
            '021': False,
            '022': True,
            '023': False,
            '024': True,
            '025': False,
            '026': True,
            '027': False,
            '028': False,
            '029': True,
            '030': False,
            '031': True,
            '032': True,
            '033': False,
            '034': True,
            '035': False,
            '036': True,
            '037': False,
            '038': True,
            '039': False,
            '040': False,
            '041': True,
            '042': True,
            '043': True,
            '044': False,
            '045': True,
            '046': False,
            '047': False,
            '048': False,
            '049': False}
vuno_lut = {0: 'Unknown',
            1: 'Left-Cerebral-Exterior',
            2: 'Left-Cerebral-White-Matter',
            3: 'Left-Cerebral-Cortex',
            4: 'Left-Lateral-Ventricle',
            5: 'Left-Inf-Lat-Vent',
            6: 'Left-Cerebellum-Exterior',
            7: 'Left-Cerebellum-White-Matter',
            8: 'Left-Cerebellum-Cortex',
            9: 'Left-Thalamus',
            10: 'Left-Thalamus-Proper*',
            11: 'Left-Caudate',
            12: 'Left-Putamen',
            13: 'Left-Pallidum',
            14: '3rd-Ventricle',
            15: '4th-Ventricle',
            16: 'Brain-Stem',
            17: 'Left-Hippocampus',
            18: 'Left-Amygdala',
            19: 'Left-Insula',
            20: 'Left-Operculum',
            21: 'Line-1',
            22: 'Line-2',
            23: 'Line-3',
            24: 'CSF',
            25: 'Left-Lesion',
            26: 'Left-Accumbens-area',
            27: 'Left-Substancia-Nigra',
            28: 'Left-VentralDC',
            29: 'Left-undetermined',
            30: 'Left-vessel',
            31: 'Left-choroid-plexus',
            32: 'Left-F3orb',
            33: 'Left-lOg',
            34: 'Left-aOg',
            35: 'Left-mOg',
            36: 'Left-pOg',
            37: 'Left-Stellate',
            38: 'Left-Porg',
            39: 'Left-Aorg',
            40: 'Right-Cerebral-Exterior',
            41: 'Right-Cerebral-White-Matter',
            42: 'Right-Cerebral-Cortex', 
            43: 'Right-Lateral-Ventricle', 
            44: 'Right-Inf-Lat-Vent',
            45: 'Right-Cerebellum-Exterior',
            46: 'Right-Cerebellum-White-Matter',
            47: 'Right-Cerebellum-Cortex',
            48: 'Right-Thalamus',
            49: 'Right-Thalamus-Proper*',
            50: 'Right-Caudate',
            51: 'Right-Putamen',
            52: 'Right-Pallidum',
            53: 'Right-Hippocampus',
            54: 'Right-Amygdala',
            55: 'Right-Insula',
            56: 'Right-Operculum',
            57: 'Right-Lesion',
            58: 'Right-Accumbens-area',
            59: 'Right-Substancia-Nigra',
            60: 'Right-VentralDC',
            61: 'Right-undetermined',
            62: 'Right-vessel',
            63: 'Right-choroid-plexus',
            64: 'Right-F3orb',
            65: 'Right-lOg',
            66: 'Right-aOg',
            67: 'Right-mOg',
            68: 'Right-pOg',
            69: 'Right-Stellate',
            70: 'Right-Porg', 
            71: 'Right-Aorg',
            72: '5th-Ventricle',
            73: 'Left-Interior',
            74: 'Right-Interior',
            77: 'WM-hypointensities',
            78: 'Left-WM-hypointensities',
            79: 'Right-WM-hypointensities',
            80: 'non-WM-hypointensities',
            81: 'Left-non-WM-hypointensities',
            82: 'Right-non-WM-hypointensities',
            83: 'Left-F1',
            84: 'Right-F1',
            85: 'Optic-Chiasm',
            192: 'Corpus_Callosum',
            86: 'Left_future_WMSA',
            87: 'Right_future_WMSA',
            88: 'future_WMSA',
            96: 'Left-Amygdala-Anterior',
            97: 'Right-Amygdala-Anterior',
            98: 'Dura',
            100: 'Left-wm-intensity-abnormality',
            101: 'Left-caudate-intensity-abnormality',
            102: 'Left-putamen-intensity-abnormality',
            103: 'Left-accumbens-intensity-abnormality',
            104: 'Left-pallidum-intensity-abnormality',
            105: 'Left-amygdala-intensity-abnormality',
            106: 'Left-hippocampus-intensity-abnormalit',
            107: 'Left-thalamus-intensity-abnormality',
            108: 'Left-VDC-intensity-abnormality',
            109: 'Right-wm-intensity-abnormality',
            110: 'Right-caudate-intensity-abnormality',
            111: 'Right-putamen-intensity-abnormality',
            112: 'Right-accumbens-intensity-abnormality',
            113: 'Right-pallidum-intensity-abnormality',
            114: 'Right-amygdala-intensity-abnormality',
            115: 'Right-hippocampus-intensity-abnormali',
            116: 'Right-thalamus-intensity-abnormality',
            117: 'Right-VDC-intensity-abnormality',
            118: 'Epidermis',
            119: 'Conn-Tissue',
            120: 'SC-Fat-Muscle',
            121: 'Cranium',
            122: 'CSF-SA',
            123: 'Muscle',
            124: 'Ear',
            125: 'Adipose',
            126: 'Spinal-Cord',
            127: 'Soft-Tissue',
            128: 'Nerve',
            129: 'Bone',
            130: 'Air',
            131: 'Orbital-Fat',
            132: 'Tongue',
            133: 'Nasal-Structures',
            134: 'Globe',
            135: 'Teeth',
            136: 'Left-Caudate-Putamen',
            137: 'Right-Caudate-Putamen',
            138: 'Left-Claustrum',
            139: 'Right-Claustrum',
            140: 'Cornea',
            142: 'Diploe',
            143: 'Vitreous-Humor',
            144: 'Lens',
            145: 'Aqueous-Humor',
            146: 'Outer-Table',
            147: 'Inner-Table',
            148: 'Periosteum',
            149: 'Endosteum',
            150: 'R-C-S',
            151: 'Iris',
            152: 'SC-Adipose-Muscle',
            153: 'SC-Tissue',
            154: 'Orbital-Adipose',
            155: 'Left-IntCapsule-Ant',
            156: 'Right-IntCapsule-Ant',
            157: 'Left-IntCapsule-Pos',
            158: 'Right-IntCapsule-Pos',
            159: 'Left-Cerebral-WM-unmyelinated',
            160: 'Right-Cerebral-WM-unmyelinated',
            161: 'Left-Cerebral-WM-myelinated',
            162: 'Right-Cerebral-WM-myelinated',
            163: 'Left-Subcortical-Gray-Matter',
            164: 'Right-Subcortical-Gray-Matter',
            165: 'Skull',
            166: 'Posterior-fossa',
            167: 'Scalp',
            168: 'Hematoma',
            169: 'Left-Basal-Ganglia',
            176: 'Right-Basal-Ganglia',
            170: 'brainstem',
            171: 'DCG',
            172: 'Vermis',
            173: 'Midbrain',
            174: 'Pons',
            175: 'Medulla',
            177: 'Vermis-White-Matter',
            178: 'SCP',
            179: 'Floculus',
            180: 'Left-Cortical-Dysplasia',
            181: 'Right-Cortical-Dysplasia',
            182: 'CblumNodulus',
            193: 'Left-hippocampal_fissure',
            194: 'Left-CADG-head',
            195: 'Left-subiculum',
            196: 'Left-fimbria',
            197: 'Right-hippocampal_fissure',
            198: 'Right-CADG-head',
            199: 'Right-subiculum',
            200: 'Right-fimbria',
            201: 'alveus',
            202: 'perforant_pathway',
            203: 'parasubiculum',
            204: 'presubiculum',
            205: 'subiculum',
            206: 'CA1',
            207: 'CA2',
            208: 'CA3',
            209: 'CA4',
            210: 'GC-DG',
            211: 'HATA',
            212: 'fimbria',
            213: 'lateral_ventricle',
            214: 'molecular_layer_HP',
            215: 'hippocampal_fissure',
            216: 'entorhinal_cortex',
            217: 'molecular_layer_subiculum',
            218: 'Amygdala',
            219: 'Cerebral_White_Matter',
            220: 'Cerebral_Cortex',
            221: 'Inf_Lat_Vent',
            222: 'Perirhinal',
            223: 'Cerebral_White_Matter_Edge',
            224: 'Background',
            225: 'Ectorhinal',
            226: 'HP_tail',
            250: 'Fornix',
            251: 'CC_Posterior',
            252: 'CC_Mid_Posterior',
            253: 'CC_Central',
            254: 'CC_Mid_Anterior',
            255: 'CC_Anterior',
            256: 'Voxel-Unchanged',
            257: 'CSF-ExtraCerebral',
            258: 'Head-ExtraCerebral',
            259: 'SkullApprox',
            260: 'BoneOrAir',
            261: 'PossibleFluid',
            262: 'Sinus',
            263: 'Left-Eustachian',
            264: 'Right-Eustachian',
            1000: 'lh-unknown',
            1001: 'lh-bankssts',
            1002: 'lh-caudalanteriorcingulate',
            1003: 'lh-caudalmiddlefrontal',
            1004: 'lh-corpuscallosum',
            1005: 'lh-cuneus',
            1006: 'lh-entorhinal',
            1007: 'lh-fusiform',
            1008: 'lh-inferiorparietal',
            1009: 'lh-inferiortemporal',
            1010: 'lh-isthmuscingulate',
            1011: 'lh-lateraloccipital',
            1012: 'lh-lateralorbitofrontal',
            1013: 'lh-lingual',
            1014: 'lh-medialorbitofrontal',
            1015: 'lh-middletemporal',
            1016: 'lh-parahippocampal',
            1017: 'lh-paracentral',
            1018: 'lh-parsopercularis',
            1019: 'lh-parsorbitalis',
            1020: 'lh-parstriangularis',
            1021: 'lh-pericalcarine',
            1022: 'lh-postcentral',
            1023: 'lh-posteriorcingulate',
            1024: 'lh-precentral',
            1025: 'lh-precuneus',
            1026: 'lh-rostralanteriorcingulate',
            1027: 'lh-rostralmiddlefrontal',
            1028: 'lh-superiorfrontal',
            1029: 'lh-superiorparietal',
            1030: 'lh-superiortemporal',
            1031: 'lh-supramarginal',
            1032: 'lh-frontalpole',
            1033: 'lh-temporalpole',
            1034: 'lh-transversetemporal',
            1035: 'lh-insula',
            2000: 'rh-unknown',
            2001: 'rh-bankssts',
            2002: 'rh-caudalanteriorcingulate',
            2003: 'rh-caudalmiddlefrontal',
            2004: 'rh-corpuscallosum',
            2005: 'rh-cuneus',
            2006: 'rh-entorhinal',
            2007: 'rh-fusiform',
            2008: 'rh-inferiorparietal',
            2009: 'rh-inferiortemporal',
            2010: 'rh-isthmuscingulate',
            2011: 'rh-lateraloccipital',
            2012: 'rh-lateralorbitofrontal',
            2013: 'rh-lingual',
            2014: 'rh-medialorbitofrontal',
            2015: 'rh-middletemporal',
            2016: 'rh-parahippocampal',
            2017: 'rh-paracentral',
            2018: 'rh-parsopercularis',
            2019: 'rh-parsorbitalis',
            2020: 'rh-parstriangularis',
            2021: 'rh-pericalcarine',
            2022: 'rh-postcentral',
            2023: 'rh-posteriorcingulate',
            2024: 'rh-precentral',
            2025: 'rh-precuneus',
            2026: 'rh-rostralanteriorcingulate',
            2027: 'rh-rostralmiddlefrontal',
            2028: 'rh-superiorfrontal',
            2029: 'rh-superiorparietal',
            2030: 'rh-superiortemporal',
            2031: 'rh-supramarginal',
            2032: 'rh-frontalpole',
            2033: 'rh-temporalpole',
            2034: 'rh-transversetemporal',
            2035: 'rh-insula'}