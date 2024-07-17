#!/bin/sh

python scripts/train_rnn.py -x 4 --early-stopping --nepoch 100 --tokenize -s results/rnnscores.csv data/*.csv
python scripts/test_rnn_voting.py --model rnn.pt --tokenize data/patches/*.csv
python scripts/train_rnn_crf.py --freeze -x 4 --nepoch 1 -i rnn.pt --tokenize -s results/rnncrffreezealltrans1.csv data/*.csv
python scripts/test_rnn_voting.py --model rnn_crf.pt --tokenize data/patches/*.csv
