#!/bin/sh

black alexi scripts test
isort --profile black alexi scripts test
flake8 alexi scripts test
mypy alexi scripts test
