#!/bin/bash

# Blank install script for django 1.10
pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org bcrypt docutils geoip2 "Jinja2>=2.7" numpy Pillow PyYAML pytz selenium sqlparse tblib coverage cosmic-ray python3-memcached

python -m pip install -e .
