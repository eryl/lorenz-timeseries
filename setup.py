# coding: utf-8

from setuptools import setup

def readme():
    with open('README.txt') as f:
        return f.read()

setup(name='lorenz',
      version='0.1',
      description='Time series synthetic data derived from the Lorenz system',
      url='https://github.com/eryl/lorenz-timeseries',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['lorenz'],
      install_requires=['numpy', 'h5py'],
      dependency_links=[],
      zip_safe=False)
