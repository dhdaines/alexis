#!/bin/sh

set -e

C1="0.01 0.05 0.1 0.2 0.5 1.0"
C2="0.01 0.05 0.1 0.2 0.5 1.0"
FEATURES="literal delta vl vsl"
OUTDIR=grid-$(date +%Y%m%d-%H:%M)

mkdir -p $OUTDIR
for f in $FEATURES; do
    for c1 in $C1; do
	for c2 in $C2; do
	    echo "Training $f L1 $c1 L2 $c2"
	    python scripts/train_crf.py -x 0 --c1 $c1 --c2 $c2 --features $f \
		   --scores $OUTDIR/scores-${f}-${c1}-${c2}.csv data/*.csv 2>&1
	done
    done
done
