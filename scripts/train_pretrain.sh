#!/bin/sh

set -e
hs=160
mf=5

# Extract features
python scripts/train_rnn.py -o feats.pt --hidden-size $hs \
       --nepoch 1 --min-feat $mf data/*.csv

# Pre-train on lq+data
python scripts/train_rnn.py -o lqfeats.pt --hidden-size $hs --nepoch 5 -i feats.pt \
       data/*.csv legisquebec/train/*.csv

# Fine-tune on data only
python scripts/train_rnn.py -i lqfeats.pt -x 4 --hidden-size $hs \
       --early-stopping --min-epochs 20 --nepoch 100 \
       -s rnn-pretrain-feats-$hs.csv data/*.csv

# Fine-tune again to validate
python scripts/train_rnn.py -i lqfeats.pt -x 4 --hidden-size $hs \
       --early-stopping --min-epochs 20 --nepoch 100 \
       -s rnn-pretrain-feats-$hs-rerun.csv data/*.csv
