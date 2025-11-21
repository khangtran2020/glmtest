#!/bin/bash

# Blank install script for pytest 6.3
pip install attrs==23.1.0 iniconfig==2.0.0 packaging==23.1 setuptools==68.0.0 pluggy==0.13.1 py==1.11.0 toml==0.10.2
pip install coverage cosmic-ray

pip install -e .
