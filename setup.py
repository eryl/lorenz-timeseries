# coding: utf-8

from setuptools import setup

def readme():
    with open('README.txt') as f:
        return f.read()

setup(name='vindel',
      version='0.1',
      description='VindEl probabalistic models example',
      url='https://github.com/eryl/vindel',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['vindel'],
      install_requires=['numpy', 'tensorflow'],
      dependency_links=[],
      zip_safe=False)
