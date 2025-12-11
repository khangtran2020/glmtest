#!/bin/bash

# Blank install script for sympy 1.13
pip install mpmath pytest pytest-xdist pytest-timeout pytest-split pytest-doctestplus hypothesis flake8 flake8-comprehensions coverage cosmic-ray
pip install mpmath==1.3.0

pip install -e .