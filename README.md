# HFT-Backtesting
![](https://github.com/evgerher/hft-backtesting/workflows/test/badge.svg)

This repository contains implementation of a framework for hft (tick-data) backtesting.  
You may find a lot of examples in `notebooks` and `tests`, it will be described later.  

## Data Source

Backtesting operates on top of historical data of specific format.  
Example of such data can be found in `tests/resources/orderbook/`.  
If you wish to collect own data, please refer to `hft.dataloader.loader.py`
- This module allows to download and store data in Clickhouse db  
- Required data format is simple CSV extracted from database (more details in fields initialization)  

## Example

In order to run a smallest example, you must initialize `reader`, `strategy`, `backtester`, and your `metrics`.  
`output` is optional, it allows to store results during backtest evaluation and visualize its performance.  

```python
reader = TimeLimitedReader('orderbook_10_03_20.csv.gz', 
                         limit_time='30 min',
                         trades_file='trades_10_03_20.csv.gz',
                         nrows=500000)

delay = 1000 # millisec
target_contracts = 3000
filter_depth=4

strategy = GatlingMM(target_contracts, filter_depth=filter_depth)
backtester = backtest.Backtest(reader, strategy, delay=delay)
```
