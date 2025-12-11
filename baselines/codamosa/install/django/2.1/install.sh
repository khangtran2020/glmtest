#!/bin/bash

# Blank install script for django 2.1
pip install "argon2-cffi>=16.1.0" bcrypt docutils geoip2 "jinja2>=2.9.2" numpy Pillow "python-memcached>=1.59" pytz PyYAML selenium sqlparse tblib coverage cosmic-ray pylibmc

python -m pip install -e .