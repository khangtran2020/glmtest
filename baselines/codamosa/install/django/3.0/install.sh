#!/bin/bash

# Blank install script for django 3.0

pip install "asgiref~=3.2" "argon2-cffi>=16.1.0" bcrypt docutils geoip2 "jinja2>=2.9.2" numpy "Pillow!=5.4.0" pylibmc "python-memcached>=1.59" pytz pywatchman PyYAML selenium "sqlparse>=0.2.2" "tblib>=1.5.0" coverage cosmic-ray

python -m pip install -e . 
