#!/bin/bash

# Make sure naming by Vuno is the same as naming by FreeSurfer

for FILE in $(ls /home/mgpoirot/lood_storage/divi/Projects/depredict/cohorten/sleeep/mask_vuno/*/*.nii.gz)
do
	DNAME="$(dirname $FILE)"
	FNAME="$(basename $FILE | cut -c 5-)" 
	mv $FILE $DNAME/$FNAME	
done 

