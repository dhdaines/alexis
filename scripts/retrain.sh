#!/bin/sh

set -e

python scripts/train_crf.py --features text+layout --outfile alexi/models/crf.vl.joblib.gz data/*.csv
python scripts/train_crf.py --features text+layout+structure --outfile alexi/models/crf.joblib.gz data/*.csv
