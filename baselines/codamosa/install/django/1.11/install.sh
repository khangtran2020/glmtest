#!/bin/bash

# Blank install script for django 1.11
pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org "argon2-cffi>=16.1.0" bcrypt docutils geoip2 "jinja2>=2.9.2" numpy Pillow PyYAML pytz selenium sqlparse tblib coverage cosmic-ray python3-memcached pylibmc

python -m pip install -e .