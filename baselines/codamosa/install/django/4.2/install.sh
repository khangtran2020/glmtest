#!/bin/bash

# Blank install script for django 4.2
pip install aiosmtpd "asgiref>=3.6.0" "argon2-cffi>=19.2.0" backports.zoneinfo bcrypt black docutils geoip2 "jinja2>=2.11.0" numpy "Pillow>=6.2.1" pylibmc "pymemcache>=3.4.0" pytz pywatchman PyYAML "redis>=3.4.0" selenium "sqlparse>=0.3.1" "tblib>=1.5.0" tzdata colorama coverage cosmic-ray

python -m pip install -e . 