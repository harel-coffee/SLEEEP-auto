import os
from glob import glob
import nibabel as nib
from shutil import copyfile

regex = r'/data/projects/depredict/sleeep/segs_{}/{}/r{}.nii'

del_all = False

for file in glob(regex.format(*'*'*3)):
	orig_fname = file.split(os.sep)[-1][1:]
	orig = os.path.join(os.path.dirname(file), orig_fname)
	if nib.load(orig).shape != nib.load(file).shape:
		print(os.path.basename(file), '-->', os.path.basename(orig))
		copyfile(file, orig)
		os.remove(file)
	else:
		print(os.sep.join(file.split(os.sep)[-3:]), '-->')
		if not del_all:
			response = input('Delete {}?'.format(os.path.basename(file)))
		if response == 'YES!':
			del_all = True
		if response == 'y' or del_all:
			print('Deleted', file)
			os.remove(file)
		else: 
			print('Rejected deletion')
