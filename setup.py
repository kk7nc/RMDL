from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='RMDL',
    version='1.0.1',
    url='https://github.com/kk7nc/RMDL',
    license='GNU General Public License v3.0',
    author='Kamran Kowsari',
    author_email='kk7nc@virginia.edu',
    description='RMDL: Random Multimodel Deep Learning for Classification',
    long_description=read('README.md'),
    install_requires=required
)


import io
import os


import numpy
from setuptools import Extension
from setuptools import setup, find_packages

__author__ = 'Kamran Kowsari <kowsari.net>'


def readme():
    with open('README.md') as f:
        return f.read()



def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()


setup(
    name='RMDL',
    version='1.0.0',
    description='RMDL: Random Multimodel Deep Learning for Classification',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    url='https://github.com/kk7nc/RMDL',
    author='Kamran Kowsari',
    author_email='kk7nc@virginia.edu',
    packages=['RMDL'] + find_packages('RMDL'),
    install_requires=[
        'matplotlib==2.1.2',
        'numpy==1.12.1',
        'pandas==0.22.0',
        'scipy',
        'tensorflow-gpu',
        'keras==2.0.9',
        'scikit-learn==0.19.0',
        'nltk==3.2.4'
    ],
    include_package_data=False
)

