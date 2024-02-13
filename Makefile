.PHONY: venv tests check-black check-flake lint format examples
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

build:
	${PYTHON} -m build
