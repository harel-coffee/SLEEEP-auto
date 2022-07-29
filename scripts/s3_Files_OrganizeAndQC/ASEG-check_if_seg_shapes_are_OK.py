from glob import glob
import nibabel as nib
import os
regex = '/data/projects/depredict/sleeep/{}_{}/{}/{}'

for tool in ['samseg', 'vuno', 'fastsurfer', 'freesurfer']:
	files = glob(regex.format('vols', 'MEMPRAGE', '*', '*.nii.gz'))
	if not len(files):
		print('No files found...')

	for file in files:
		pt, ses = file.split(os.sep)[-2:]
		memprage_shape = nib.load(file).shape
		#print('\nvol', pt, ses, nib.load(file).shape)
		segments_shape = nib.load(regex.format('segs', tool,  pt, ses)).shape
		if memprage_shape != segments_shape:
			print(os.path.join('...', tool, pt, ses), segments_shape)
		else:
			print('OK', file)
		
