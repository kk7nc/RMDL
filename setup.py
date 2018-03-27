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
