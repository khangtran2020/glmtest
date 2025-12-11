#!/bin/bash

# Blank install script for django 4.1
pip install aiosmtpd "asgiref>=3.4.1" "argon2-cffi>=16.1.0" backports.zoneinfo bcrypt black docutils geoip2 "jinja2>=2.9.2" numpy "Pillow>=6.2.0" pylibmc "pymemcache>=3.4.0" pytz pywatchman PyYAML "redis>=3.0.0" selenium "sqlparse>=0.2.2" "tblib>=1.5.0" tzdata colorama coverage cosmic-ray

python -m pip install -e . 