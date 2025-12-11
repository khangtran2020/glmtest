#!/bin/bash

# Blank install script for matplotlib 3.5
pip install cairocffi "contourpy>=1.0.1" "cycler>=0.10.0" "fonttools>=4.22.0" "kiwisolver>=1.0.1" "numpy>=1.19" "pillow>=6.2" pygobject pyparsing pyqt "python-dateutil>=2.1" setuptools setuptools_scm wxpython colorspacious graphviz ipython ipywidgets "numpydoc>=0.8" packaging pydata-sphinx-theme "sphinx>=1.8.1,!=2.0.0" sphinx-copybutton "sphinx-gallery>=0.10" sphinx-design mpl-sphinx-theme sphinxcontrib-svg2pdfconverter pikepdf coverage "flake8>=3.8" "flake8-docstrings>=1.4.0" gtk3 ipykernel "nbconvert[execute]!=6.0.0,!=6.0.1" "nbformat!=5.0.0,!=5.0.1" "pandas!=0.25.0" psutil pre-commit "pydocstyle>=5.1.0" "pytest!=4.6.0,!=5.4.0" pytest-cov pytest-rerunfailures pytest-timeout pytest-xdist tornado pytz


pip install coverage cosmic-ray
python -m pip install -e .