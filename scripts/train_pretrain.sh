#!/bin/sh

set -e
: ${hs:=160}
: ${mf:=5}

# Pre-train on lq
python scripts/train_rnn.py -o lqfeats.pt --hidden-size $hs --nepoch 1 \
       legisquebec/train/*.csv

# Fine-tune on data only
python scripts/train_rnn.py -i lqfeats.pt -x 4 --hidden-size $hs \
       --early-stopping --nepoch 100 \
       -s rnn-pretrain-feats-$hs-$mf.csv data/*.csv

# Fine-tune again to validate
python scripts/train_rnn.py -i lqfeats.pt -x 4 --hidden-size $hs \
       --early-stopping --nepoch 100 \
       -s rnn-pretrain-feats-$hs-$mf-rerun.csv data/*.csv
