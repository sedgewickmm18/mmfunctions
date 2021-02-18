#!/usr/bin/env python
import ast

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    f.close()

with open('mmfunctions/__init__.py') as f:
    version_ = f.read()
    exec(version_)
    f.close()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mmfunctions',
    version=__version__,
    author='Markus Müller',
    author_email='sedgewickmm18@googlemail.com',
    description='Helper package to be used in conjunction with the Maximo Asset Manager pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sedgewickmm18/mmfunctions',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
