import io
import os


import numpy
from setuptools import Extension
from setuptools import setup, find_packages
from os import path

__author__ = 'Kamran Kowsari'



here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
def readfile(file):
    with open(path.join(here, file)) as f:
        long_description = f.read()
    return long_description


setup(
    name='RMDL',
    version='1.0.8',
    description='RMDL: Random Multimodel Deep Learning for Classification',
    long_description=readfile('README.rst'),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Editors :: Text Processing',
    ],
    url='https://github.com/kk7nc/RMDL',
    author='Kamran Kowsari',
    author_email='kk7nc@virginia.edu',
    install_requires=[
        'matplotlib>=2.1.2',
        'numpy>=1.12.1',
        'pandas>=0.22.0',
        'scipy',
        'tensorflow',
        'keras>=2.0.9',
        'scikit-learn>=0.19.0',
        'nltk>=3.2.4'
    ],
    packages=find_packages()
)

