#!/bin/sh

for n in 1 2 3; do
    for features in text layout text+ text+layout structure text+layout+structure; do
        python scripts/train_crf.py --features $features -x 4 data/*.csv -s results/$features-x4.csv
    done
done
