import datetime
import unittest
from typing import Dict, Tuple, Union, Deque, List

from backtesting import backtest
from backtesting.data import OrderRequest
from backtesting.output import TestOutput
from backtesting.readers import ListReader
from backtesting.strategy import Strategy
from metrics.metrics import VWAP_volume, TimeMetric, InstantMetric
from utils.data import OrderBook, Trade
import numpy as np


class StrategyTest(unittest.TestCase):
  def test_balance(self):
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 200),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300), 'Sell', 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300),
                np.array([9.5, 9.0, 8.5]), np.array([900, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 400),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400),
                np.array([9.0, 8.5, 8.0]), np.array([100, 100, 140]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 2000),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 1300),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 1300),
    ])

    ##########################################################
    class SimpleStrategy(Strategy):
      def __init__(self, instant_metrics: List[InstantMetric], time_metrics):
        super().__init__(instant_metrics, time_metrics=time_metrics)
        self.idx = 0

      def define_orders(self, row: Union[Trade, OrderBook],
                    memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
                    snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
                    trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
                    trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]):
        item = []
        if self.idx == 0:
          item = [OrderRequest.create_bid(9.5, 450, 'test', reader[0].timestamp)]
        elif self.idx == 4:
          item = [OrderRequest.create_bid(9.5, 200, 'test', reader[3].timestamp)]

        self.idx += 1
        return item

    ###########################################################

    time_metrics = [TimeMetric(callables, 60), TimeMetric(callables, 30)]
    simulation = SimpleStrategy(instant_metrics, time_metrics=time_metrics)

    output = TestOutput(instant_metric_names=instant_metric_names,
                        time_metric_names=[metric.metric_names for metric in time_metrics])



    backtester = backtest.Backtest(reader, simulation, output)

    initial_balance = dict(simulation.balance)

    backtester._process_event(reader[0])
    self.assertEqual(initial_balance['USD'] - 450, simulation.balance['USD'])

    for event in reader[1:7]:
      backtester._process_event(event)

    self.assertEqual(initial_balance['USD'] - 650, simulation.balance['USD'])

    for event in reader[7:-1]:
      backtester._process_event(event)

    self.assertEqual(initial_balance['USD'] - 650, simulation.balance['USD'])
    self.assertAlmostEqual(450.0 / 9.5, simulation.balance['test'], delta=1e-3)

    backtester._process_event(reader[-1])
    self.assertEqual(initial_balance['USD'] - 650, simulation.balance['USD'])
    self.assertAlmostEqual(650.0 / 9.5, simulation.balance['test'], delta=1e-3)
