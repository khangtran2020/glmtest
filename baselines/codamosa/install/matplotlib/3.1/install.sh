#!/bin/bash

# Blank install script for matplotlib 3.1
pip install coverage "pytest!=4.6.0" pytest-cov pytest-rerunfailures pytest-timeout pytest-xdist python-dateutil tornado

pip install coverage cosmic-ray
python -m pip install -e .