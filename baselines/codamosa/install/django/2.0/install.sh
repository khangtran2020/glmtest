#!/bin/bash

# Blank install script for django 2.0
pip install "argon2-cffi>=16.1.0" bcrypt docutils geoip2 "jinja2>=2.9.2" numpy Pillow pytz PyYAML selenium sqlparse tblib coverage cosmic-ray "python-memcached>=1.59" pylibmc

python -m pip install -e .