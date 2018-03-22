from setuptools import setup
from setuptools import find_packages


setup(name='RMDL',
      version='1.0.0',
      description='Random Multimodel Deep Learning for Classification',
      author='Kamran Kowsari',
      author_email='kk7nc@virginia.edu',
      url='https://github.com/kk7nc/RMDL',
      download_url='',
      license='GPL-3.0',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml'],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov',
                    'pandas',
                    'requests'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GPL-3.0 License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
