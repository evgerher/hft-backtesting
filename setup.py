from setuptools import setup, find_packages

setup(
  name='backtesting-hft',
  version='0.5.0',
  packages=find_packages(include=['hft', 'hft.*']),
  url='',
  license='',
  author='Evgeny Sorokin',
  author_email='evgeniy.inpl.sorokin@gmail.com',
  description='Extendible backtesting platform for HFT based on orderbooks and trades from Bitmex'
)
