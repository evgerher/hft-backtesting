from setuptools import setup

setup(
  name='backtesting-hft',
  version='1.0.0',
  packages=['utils', 'metrics', 'dataloader', 'dataloader.callbacks', 'dataloader.callbacks.kdb',
            'dataloader.callbacks.clickhouse', 'strategies', 'backtesting'],
  package_dir={'': 'project'},
  url='',
  license='',
  author='Evgeny Sorokin',
  author_email='evgeniy.inpl.sorokin@gmail.com',
  description='Extendible backtesting platform for HFT based on snapshots and orderbooks from Bitmex'
)
