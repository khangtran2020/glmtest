#!/bin/bash

# Blank install script for matplotlib 3.0
pip install codecov coverage cycler numpy pillow pyparsing pytest pytest-cov pytest-faulthandler pytest-rerunfailures pytest-timeout pytest-xdist python-dateutil tornado tox

pip install coverage cosmic-ray
python -m pip install -e .