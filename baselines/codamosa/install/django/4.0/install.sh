#!/bin/bash

# Blank install script for django 4.0
pip install "asgiref>=3.3.2" "argon2-cffi>=16.1.0" backports.zoneinfo bcrypt docutils geoip2 "jinja2>=2.9.2" numpy "Pillow>=6.2.0" pylibmc "pymemcache>=3.4.0" "python-memcached>=1.59" pytz pywatchman PyYAML "redis>=3.0.0" selenium "sqlparse>=0.2.2" "tblib>=1.5.0" tzdata colorama coverage cosmic-ray

python -m pip install -e . 