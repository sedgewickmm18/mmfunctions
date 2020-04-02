#!/usr/bin/env python

from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mmfunctions',
    version='0.0.3',
    packages=find_packages(),
    install_requires=requirements
)
