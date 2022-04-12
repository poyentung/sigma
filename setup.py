#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
__version__ = '0.1.0'
    
setup(
    name='sigmapy',
    version=__version__,
    description="spectral interpretation using gaussian mixtures and autoencoder ",
    author='Po-Yen Tung',
    author_email='pyt21@cam.ac.uk',
    license='GNU GPLv3',
    url="https://github.com/poyentung/sigma",
    long_description=open("README.md").read(),
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
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    install_requires=[
        "torch          >= 1.10.0+cu111",
        "hyperspy       >= 1.6.5",  
        "ipywidgets",
        "lmfit          >= 0.9.12",
        "matplotlib     >= 3.2.2",  # 3.2.1 failed
        "numba",
        "numpy          >=1.19.5",
        "scikit-learn   >= 1.0.2",  # reason unknown
        "scipy",
        "tqdm           >=4.62.3",
        "seaborn        >=0.11.2",
        "plotly         >=4.4.1",
        "altair         >=4.2.0"
        
    ],
    python_requires=">=3.7",
    package_data={
        "": ["LICENSE", "README.md"],
        "sigma": ["*.py"],
    },
    entry_points={"hyperspy.extensions": ["pyxem = pyxem"]},
)
