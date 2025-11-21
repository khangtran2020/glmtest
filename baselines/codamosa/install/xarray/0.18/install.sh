#!/bin/bash

# Blank install script for xarray 0.18
pip install aiobotocore boto3 bottleneck cartopy cdms2 cfgrib cftime dask distributed "fsspec!=2021.7.0" h5netcdf h5py hdf5 hypothesis iris lxml matplotlib-base nc-time-axis netcdf4 numba numexpr numpy pandas pint pooch pre-commit pseudonetcdf pydap pytest pytest-cov pytest-env pytest-xdist rasterio scipy seaborn setuptools sparse toolz zarr numbagg

pip install numpy==1.25.2 packaging==23.1 pandas==1.5.3 pytest==8.1.1 python-dateutil==2.8.2 pytz==2023.3 six==1.16.0

pip install -e .
pip install coverage cosmic-ray