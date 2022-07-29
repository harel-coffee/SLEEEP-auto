#!/bin/bash

# March 2021, This code runs FastSurfer ASEG in bulk on all files
# Runs many in parallel (background)
# m.g.poirot

# Must supply an existing T1 input (conformed, full head) via --t1 (absolute path and name).
FILES=/data/projects/depredict/sleeep/freesurfer_working_directory/*/*/mri/orig.mgz

datadir=$(echo "$FILES" | cut -d/ -f-6)

fastsurferdir=/data/projects/depredict/sleeep/scripts/FastSurfer/my_fastsurfer_analysis

for FILE in $FILES
do
	SID=$(echo "$FILE" | cut -d/ -f7)
	DATE=$(echo "$FILE" | cut -d/ -f8)
	./run_fastsurfer.sh --t1 $datadir/$SID/$DATE/mri/orig.mgz \
                    --sid "${SID}/${DATE}" --sd $fastsurferdir \
                    --parallel --threads 64 --py python3.7 --batch 1
done



