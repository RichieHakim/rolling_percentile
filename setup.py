import setuptools
from distutils.core import setup

setup(
    name='rp',
    version='0.2.0',
    author='Richard Hakim',
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.7'],
    packages=['rp'],
    long_description=open('README.md').read(),
    url='https://github.com/RichieHakim/rolling_percentile',
)