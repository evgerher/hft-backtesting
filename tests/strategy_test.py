import datetime
import unittest

from hft.backtesting import backtest
import numpy as np

from hft.backtesting.output import StorageOutput
from hft.backtesting.readers import ListReader
from hft.units.metrics.instant import VWAP_volume
from hft.units.metrics.time import TradeMetric
from hft.utils.consts import TradeSides
from hft.utils.data import OrderBook, Trade
from test_utils import TestStrategy


class StrategyTest(unittest.TestCase):

  @unittest.skip("works only alone")
  def test_balance(self):
    callables = [
      ('_trades volume_total', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('_trades length', lambda trades: len(trades))
    ]
    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 200),
                np.array([9.5, 9.0, 8.5, 8,0]), np.array([1000, 100, 100, 100]),
                np.array([10.0, 10.5, 11.0, 12.0]), np.array([100, 100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300), TradeSides.SELL, 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300),
                np.array([9.5, 9.0, 8.5, 8.0]), np.array([900, 100, 100, 100]),
                np.array([10.0, 10.5, 11.0, 12.0]), np.array([100, 100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), TradeSides.SELL, 9.5, 400),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), TradeSides.SELL, 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400),
                np.array([9.0, 8.5, 8.0, 7.0]), np.array([100, 100, 100, 200]),
                np.array([10.0, 10.5, 11.0, 12.0]), np.array([100, 100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), TradeSides.SELL, 9.0, 550),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), TradeSides.SELL, 9.0, 200),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), TradeSides.SELL, 9.0, 1300),
    ])

    time_metrics = [TradeMetric(callables, 60), TradeMetric(callables, 30)]
    simulation = TestStrategy(instant_metrics, time_metrics_trade=time_metrics, reader=reader)

    output = StorageOutput(instant_metric_names=instant_metric_names,
                           time_metric_names=[metric.metric_names for metric in time_metrics])



    backtester = backtest.Backtest(reader, simulation, output)

    initial_balance = dict(simulation.balance)

    backtester._process_event(reader[0], type(reader[0]) == OrderBook)

    for event in reader[:-2]:
      backtester._process_event(event, type(event) == OrderBook)
    self.assertEqual(initial_balance['USD'] - 550, simulation.balance['USD'])
    self.assertAlmostEqual((450 + 100) / 9.5, simulation.balance['test'], delta=1e-3)

    for event in reader[-2:]:
      backtester._process_event(event, type(event) == OrderBook)
    self.assertEqual(initial_balance['USD'] - 650, simulation.balance['USD'])
    self.assertAlmostEqual(650.0 / 9.5, simulation.balance['test'], delta=1e-3)


if __name__ == '__main__':
  unittest.main()
