#!/bin/bash

# Blank install script for django 5.0
pip install aiosmtpd "asgiref>=3.7.0" "argon2-cffi>=19.2.0" bcrypt black docutils geoip2 "jinja2>=2.11.0" numpy "Pillow>=6.2.1" pylibmc "pymemcache>=3.4.0" pywatchman PyYAML "redis>=3.4.0" "selenium>=4.8.0" "sqlparse>=0.3.1" "tblib>=1.5.0" tzdata colorama coverage cosmic-ray

python -m pip install -e . 