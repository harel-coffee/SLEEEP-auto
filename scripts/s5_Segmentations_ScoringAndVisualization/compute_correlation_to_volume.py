# this was a bit of a side project into if volume was the main cause of low dice scores

import pickle
from glob import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm 
from matplotlib import pyplot as plt
import csv
from multiprocessing import Pool, cpu_count
from time import sleep

# store a dictionary
def pickle_in(source: str) -> dict:
	with open(source, 'rb') as file:
 		return pickle.load(file)

def pickle_out(obj : dict, target: str):
	with open(target, 'wb') as file:
		pickle.dump(obj, file)

def show(subplot_n=False):
	if subplot_n:
		fig = plt.figure()
		ax = plt.subplot(2,3,n)
		plt.scatter(x,y)
		for u, v, k in zip(x, y, dice_dict[toolcombo]):
			plt.text(u, v, naming_lut[k])
		ax.set_xscale('log')
		plt.ylabel('Dice score')
		plt.xlabel('Number of voxels')
		plt.title(toolcombo)
		plt.ylim([0,1])
		plt.xlim([100,10**6])
		plt.show()	
	print('\nYOUR QUARTERLY REPORT IS READY!')
	print(toolcombo)
	for xi,yi,k in zip(x,y,dice_dict[toolcombo]):
		print(naming_lut[k], xi, yi)
	

naming_lut = pickle_in(r'/data/projects/depredict/sleeep/scripts/3vuno_lut/naming_lut.pkl')

dice_dict_path = r'/data/projects/depredict/sleeep/scripts/4dice_scoring/label_dice_dict.pkl'
regex = r'/data/projects/depredict/sleeep/segs_{}/*/*.nii.gz'

csv_file = 'dice_volume_relations.csv'

dice_dict = pickle_in(dice_dict_path)

x = []
y = []

def compute_toolcombo(toolcombo):
	written = False
	attempts = 0
	tool1, tool2 = toolcombo.split('-')
	tool1_path, tool2_path = regex.format(tool1), regex.format(tool2)

	# create a dict to store our results
	tool1_vols, tool2_vols = {}, {}
	tool1_paths, tool2_paths = glob(tool1_path), glob(tool2_path)
	for j, (tool1_file, tool2_file) in enumerate(zip(tool1_paths, tool2_paths)):
		# Clear/prepare the results dict
		for label in dice_dict[toolcombo]:
			tool1_vols[label] = []
			tool2_vols[label] = []
		
		tool1_vol = nib.load(tool1_file).get_fdata()
		tool2_vol = nib.load(tool2_file).get_fdata()
		for k, label in enumerate(dice_dict[toolcombo]):
			tool1_vols[label].append(np.sum(tool1_vol == label))
			tool2_vols[label].append(np.sum(tool2_vol == label))
			print(toolcombo,
			      str(j) + '/' + str(len(tool1_paths)),
		      	      str(k) + '/' + str(len(dice_dict[toolcombo].keys())))

	tool1_res = [np.mean(tool1_vols[label]) for label in tool1_vols]
	tool2_res = [np.mean(tool2_vols[label]) for label in tool2_vols]
	
	x.extend([np.min((a, b)) for a, b in zip(tool1_res, tool2_res)])
	y.extend([np.mean(scores) for _, scores in dice_dict[toolcombo].items()])
	
	while not written:
		try:
			with open(csv_file, 'a', encoding='utf16') as cf:
				writer = csv.writer(cf)
				writer.writerow([toolcombo])
				writer.writerow(['anatomic_name', 'volume(px)', 'dice-score'])
				for xi,yi,k in zip(x,y,dice_dict[toolcombo]):
					writer.writerow([naming_lut[k], xi, yi])
			written = True
		except:
			attempts = attempts + 1
			sleep(0.1)
			if attempts > 10:
				response = input('10 attempts failed to write results. Quit? (Yes/[No])')
			if response in 'Yesyes':
				break
			else:
				attempts = attempts - 10
				continue

if __name__ == '__main__':
	num_cores = cpu_count()
	chucksize = 1
	toolcombos = dice_dict.keys()
	with Pool(processes=num_cores) as pool:
		pool.map(compute_toolcombo, toolcombos)
	
