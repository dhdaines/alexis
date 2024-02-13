.PHONY: venv tests check-black check-flake lint format train
PACKAGE := alexi
PYTHON := venv/bin/python
PIP = venv/bin/pip

venv:
	python3 -m venv venv
	${PIP} install --upgrade pip
	${PIP} install -r requirements-dev.txt
	${PIP} install -e .

tests:
	${PYTHON} -m pytest
	${PYTHON} -m coverage html

check-black:
	${PYTHON} -m black --check ${PACKAGE} test

check-isort:
	${PYTHON} -m isort --profile black --check-only ${PACKAGE} test

check-flake:
	${PYTHON} -m flake8 ${PACKAGE} test

check-mypy:
	${PYTHON} -m mypy --strict --implicit-reexport ${PACKAGE}

lint: check-flake check-mypy check-black check-isort

format:
	${PYTHON} -m black ${PACKAGE} test
	${PYTHON} -m isort --profile black ${PACKAGE} test

train:
	${PYTHON} scripts/train_crf.py \
		--features text+layout \
		--outfile alexi/models/crf.vl.joblib.gz \
		data/*.csv
	${PYTHON} scripts/train_crf.py \
		--features text+layout+structure \
		--outfile alexi/models/crf.joblib.gz \
		data/*.csv
	${PYTHON} scripts/train_crf_seq.py \
		--outfile alexi/models/crfseq.joblib.gz \
		data/*.csv
