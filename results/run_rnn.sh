#!/bin/sh

python scripts/train_rnn.py -x 4 --early-stopping --nepoch 100 -s results/rnnscores.csv data/*.csv
python scripts/test_rnn.py --model alexi/models/rnn.pt data/patches/*.csv
python scripts/train_rnn_crf.py --freeze -x 4 --nepoch 1 -i rnn.pt -s results/rnncrffreezealltrans.csv data/*.csv
python scripts/test_rnn.py --model alexi/models/rnn_crf.pt data/patches/*.csv
