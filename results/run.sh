#!/bin/sh

for features in text+ text+layout text+layout+structure; do
    python scripts/train_crf.py --features $features --labels bonly \
           -x 4 data/*.csv -s results/$features-x4.csv -o results/cnn_$features
    python scripts/test_crf_voting.py -m results/cnn_$features data/patches/*.csv \
           > results/$features-patches.txt
done
