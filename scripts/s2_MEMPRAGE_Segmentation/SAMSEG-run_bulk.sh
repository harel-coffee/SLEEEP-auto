#!/bin/bash	

FILES=/data/projects/depredict/sleeep/volumes_MEMPRAGE/*/*.nii.gz

ROOT=$(echo "$FILES" | cut -d/ -f1-5)/segs_samseg

for FILE in $FILES
do
	SID=$(echo "$FILE" | cut -d/ -f7-8)
	SID=$(echo "$SID" | cut -d. -f1)
        OUTDIR="${ROOT}/${SID}"

	if ! test -d "$OUTDIR"
	then
		mkdir -p $OUTDIR
	fi

	if ! test -f "$OUTDIR.nii.gz"
	then
		run_samseg --input $FILE --output $OUTDIR --threads 96
	fi
done
