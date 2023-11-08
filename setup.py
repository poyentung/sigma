#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()
__version__ = "0.2.1"
    
setup(
    name='emsigma',
    version=__version__,
    description="spectral interpretation using gaussian mixtures and autoencoder ",
    author='Po-Yen Tung',
    author_email='pyt21@cam.ac.uk',
    license='GNU GPLv3',
    url="https://github.com/poyentung/sigma",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        "hyperspectral imaging analysis",
        "energy dispersive x-ray spectroscopy",
        "scanning electron microscopy",
        "gaussain mixture",
        "autoencoder",
        "non-negative matrix factorization",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    install_requires=[
        "notebook",
        "torch          == 2.0.1",
        "hyperspy       == 1.7.5",  
        "ipywidgets     == 8.1.1",
        "lmfit          == 1.2.2",
        "matplotlib     == 3.8.0", 
        "numpy          == 1.24.4",
        "numba          == 0.57.1",
        "scikit-learn   == 1.3.0",
        "scipy",
        "umap-learn",
        "tqdm           == 4.66.1",
        "seaborn        == 0.12.2",
        "plotly         == 5.17.0",
        "altair         == 5.1.1",
        "jupyterlab     == 4.0.6", 
        "ipywidgets     == 8.1.1",
        "jupyter-dash   == 0.4.2",
        "Werkzeug       == 2.2.3",
        "Flask          == 2.2.5",
        
    ],
    python_requires=">=3.10",
    package_data={
        "": ["LICENSE", "README.md"],
        "sigma": ["*.py"],
    },
)
