#!/bin/bash	

# March 2021, This code runs FreeSurfer ASEG in bulk on all files
# Runs many in parallel (background)
#
# Freesurfer requires a specific file structure format:
#   data/sub-001/mri/oirg/001.mgz
# This is why a specific freesurfer_working_directory should be created
#
# m.g.poirot

SIDS=/data/projects/depredict/sleeep/freesurfer_working_directory/*

# for each subject...
for SID in $SIDS
do
	# for each time point within one subject...
	export SUBJECTS_DIR=$SID
	TIMES=$SID/*_*
	for TIME in $TIMES
	do
		# if no segmentation exists...
		if ! test -f $TIME/mri/aparc.DKTatlas+aseg.mgz
		then		
			T="$(basename $TIME)"
			# segment!			
			recon-all -s $T -all &
			#recon-all -s $T -autorecon3
			echo $TIME
		fi

	done
done











# September 23, 2020: This code runs FreeSurfer for all subdirectories of the SLEEEP data set

#DSLOC=/home/mgpoirot/lood_storage/divi/Projects/depredict/cohorten/sleeep/volumes/
#for FILE in $(ls -d $PWD/*/*/MEMPRAGE.nii.gz)
#do
#	ROOT="$(dirname $FILE)"
#	SID="$(basename $ROOT)"	
#	SDIR="$(dirname $ROOT)"
#	export SUBJECTS_DIR=$SDIR
#	
#	if test ! -d $ROOT/mri
#	then
#		mkdir $ROOT/mri
#	#	mkdir $ROOT/mri/orig
#	#	mri_convert $FILE $ROOT/mri/orig/001.mgz
#	#	recon-all -s $SID -all
#	#fi
#done

