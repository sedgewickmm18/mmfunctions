#!/usr/bin/env python
import ast
import os
import subprocess

from setuptools import setup, find_packages

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    f.close()

with open('mmfunctions/__init__.py') as f:
    version_ = f.read()
    exec(version_)
    f.close()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


#output = subprocess.check_output([sys.executable, '-m', 'pip', 'install', '-i', 'https://test.pypi.org/simple/', 'telemanom'])
#print(output)

from pathlib import Path
os.chdir(Path(__file__).parent.absolute())

extra_files = package_files('tsfm_public')

setup(
    name='mmfunctions',
    version=__version__,
    author='Markus Müller',
    author_email='sedgewickmm18@googlemail.com',
    description='Helper package to be used in conjunction with the Maximo Asset Manager pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sedgewickmm18/mmfunctions',
    packages=['mmfunctions', 'telemanom', 'tsfm_public'],
    package_dir={'mmfunctions':'mmfunctions',
                 'telemanom':'telemanom'},
    package_data={'': extra_files},
    #packages=find_packages(
    #    include=['mmfunctions', 'telemanom'],
    #    exclude=['custom', 'samples', 'runs', 'pipelines', 'tf-levenberg-marquardt']
    #),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=requirements
)
