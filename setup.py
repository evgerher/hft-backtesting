from setuptools import setup, find_packages

setup(
  name='backtesting-hft',
  version='1.0.0',
  packages=find_packages(include=['hft', 'hft.*']),
  # packages=['hft'],
  # package_dir={'hft': 'project'},
  url='',
  license='',
  author='Evgeny Sorokin',
  author_email='evgeniy.inpl.sorokin@gmail.com',
  description='Extendible backtesting platform for HFT based on snapshots and _orderbooks from Bitmex'
)
