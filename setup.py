#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='mmfunctions',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'dill==0.2.5',
        'urllib3>=1.24.2',
        'sqlalchemy==1.3.5',
        'numpy==1.14.5',
        'pandas==0.24.0',
        'requests>=2.20.0',
        'lxml==4.3.4',
        'ibm_db==3.0.1',
        'ibm_db_sa==0.3.3',
        'ibm-cos-sdk==2.1.3',
        'pandas==0.24.0',
        'scipy==1.1.0',
        'scikit-learn==0.19.2'
    ]
)
