from glob import glob
import os
import nibabel as nib
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
from utils import pickle_out, get_root


# Define support functions
tools = ['samseg', 'vuno', 'freesurfer', 'fastsurfer']
unique_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# compute dice score
def dice(seg1, seg2):
	return np.sum(seg1[seg2]) * 2.0 / (np.sum(seg1) + np.sum(seg2))


# format two tool names to one key: 'samseg-vuno'
def format_tools(t1, t2):
	return tools[t1] + '-' + tools[t2]


# takes two integer arrays, returns list of common labels
def get_common_labels(seg1, seg2) -> list:
	common_labels = []
	labels1, labels2 = np.unique(seg1), np.unique(seg2)
	for i in labels1:
		if i in labels2:
			common_labels.append(i)
	return common_labels


# get dice scores for one path
def get_dices(common_path):
	label_dice_dict = get_template()
	sub, ses = common_path.split(os.sep)[-2:]
	ses = ses.split(os.extsep)[0]

	# load all volumes
	label_arrays = {}
	for tool in tools:
		label_path = regex.format(tool, sub, ses)
		try:
			label_arrays[tool] = nib.load(label_path).get_fdata()
		except:
			print('FATAL ERROR WHILE LOADING:', label_path)	
			return label_dice_dict
		

	# compute dice for each tool, and for each label
	for tool1_idx, tool2_idx in unique_combinations:
		cur_tools = format_tools(tool1_idx, tool2_idx)
		seg1 = label_arrays[tools[tool1_idx]]
		seg2 = label_arrays[tools[tool2_idx]]
		label_list = get_common_labels(seg1, seg2)
		for label in label_list:
			if label == 16:
				breakpoint()
			label_dice_dict[cur_tools][label] = dice(seg1 == label, seg2 == label)
	print('Finished', common_path)
	return label_dice_dict


def merge_dicts(parent, child):
	for key in child:
		for kkey in child[key]:
			if not kkey in parent[key]:
				parent[key][kkey] = []
			parent[key][kkey].append(child[key][kkey])		
	return parent


def get_template():
	# Setup a dict to store our results in
	dice_dict_template = {}
	for tool1_idx, tool2_idx in unique_combinations:
		dice_dict_template[format_tools(tool1_idx, tool2_idx)] = {}
	return dice_dict_template


if __name__ == '__main__':
	PARALLEL = False

	# Find all files
	regex = get_root('data', r'segs_{}', '{}', '{}.nii.gz')
	common_paths = glob(regex.format('samseg', '*', '*'))
	num_cores = cpu_count()

	print(len(common_paths), 'tasks to go!')

	if PARALLEL:
		with Pool(processes=num_cores) as pool:
			results = pool.map(get_dices, common_paths)
	else:
		for path in tqdm(common_paths):
			results = get_dices(path)

	dice_dict = get_template()
	for result in results:
		dice_dict = merge_dicts(dice_dict, result)
	
	# store results
	pickle_out(dice_dict, 'label_dice_dict.pkl')
	print('Finished successfully, stored to', 'label_dice_dict.pkl')
	

