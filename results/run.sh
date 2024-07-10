#!/bin/sh

for features in text layout text+ text+layout structure text+layout+structure; do
    python scripts/train_crf.py --features $features --labels bonly -x 4 data/*.csv -s results/$features-x4.csv
done
