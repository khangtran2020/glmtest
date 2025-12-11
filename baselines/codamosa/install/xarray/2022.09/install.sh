#!/bin/bash

# Blank install script for xarray 2022.09

pip install aiobotocore boto3 bottleneck cartopy cdms2 cftime dask-core distributed flox "fsspec!=2021.7.0" h5netcdf h5py hdf5 hypothesis iris lxml matplotlib-base nc-time-axis netcdf4 numba numbagg numexpr "numpy<1.24" packaging pandas pint pooch pre-commit pseudonetcdf pydap pytest pytest-cov pytest-env pytest-xdist pytest-timeout rasterio scipy seaborn sparse toolz typing_extensions zarr


pip install numpy==1.25.2 packaging==23.1 pandas==1.5.3 pytest==8.1.1 python-dateutil==2.8.2 pytz==2023.3 six==1.16.0

pip install -e .
pip install coverage cosmic-ray