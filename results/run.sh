#!/bin/sh

for features in text text+ text+layout text+layout+structure; do
    python scripts/train_crf.py --features $features -x 4 data/*.csv -s results/$features-x4.csv
done
