import datetime
import unittest

from backtesting import backtest
from backtesting.output import StorageOutput
from backtesting.readers import ListReader
from metrics.metrics import VWAP_volume, TradeMetric
from test_utils import TestStrategy
from utils.data import OrderBook, Trade
import numpy as np


class StrategyTest(unittest.TestCase):

  @unittest.skip("works only alone")
  def test_balance(self):
    callables = [
      ('trades volume_total', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 200),
                np.array([9.5, 9.0, 8.5, 8,0]), np.array([1000, 100, 100, 100]),
                np.array([10.0, 10.5, 11.0, 12.0]), np.array([100, 100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300), 'Sell', 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300),
                np.array([9.5, 9.0, 8.5, 8.0]), np.array([900, 100, 100, 100]),
                np.array([10.0, 10.5, 11.0, 12.0]), np.array([100, 100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 400),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400),
                np.array([9.0, 8.5, 8.0, 7.0]), np.array([100, 100, 100, 200]),
                np.array([10.0, 10.5, 11.0, 12.0]), np.array([100, 100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 550),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 200),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 1300),
    ])

    time_metrics = [TradeMetric(callables, 60), TradeMetric(callables, 30)]
    simulation = TestStrategy(instant_metrics, time_metrics_trade=time_metrics, reader=reader)

    output = StorageOutput(instant_metric_names=instant_metric_names,
                           time_metric_names=[metric.metric_names for metric in time_metrics])



    backtester = backtest.Backtest(reader, simulation, output)

    initial_balance = dict(simulation.balance)

    backtester._process_event(reader[0])
    self.assertEqual(initial_balance['USD'] - 450, simulation.balance['USD'])

    for event in reader[1:7]:
      backtester._process_event(event)

    self.assertEqual(initial_balance['USD'] - 650, simulation.balance['USD'])
    self.assertAlmostEqual((450 + 100) / 9.5, simulation.balance['test'], delta=1e-3)

    for event in reader[7:]:
      backtester._process_event(event)
    self.assertAlmostEqual(650.0 / 9.5, simulation.balance['test'], delta=1e-3)


if __name__ == '__main__':
  unittest.main()
