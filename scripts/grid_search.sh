#!/bin/sh

set -e

C1="0.1 0.15 0.2 0.25 0.3"
C2="0.001 0.01 0.1"
FEATURES="text+layout text+layout+structure"
OUTDIR=grid-$(date +%Y%m%d-%H:%M)

mkdir -p $OUTDIR
for f in $FEATURES; do
    for c1 in $C1; do
	for c2 in $C2; do
	    echo "Training $f L1 $c1 L2 $c2"
	    python scripts/train_crf.py -x 0 --c1 $c1 --c2 $c2 --features $f \
		   --min-count 20 --scores $OUTDIR/scores-${f}-${c1}-${c2}.csv data/*.csv 2>&1
	done
    done
done
