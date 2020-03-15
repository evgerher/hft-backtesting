import unittest

from backtesting import readers, backtest
from backtesting.output import TestOutput
from backtesting.trade_simulation import Strategy

from metrics.filters import Filters
from metrics.metrics import *


class BacktestTest(unittest.TestCase):
  def test_run(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz', stop_after=100)
    metrics = [VWAP_depth(3),
               VWAP_volume(volumes=[int(5e5), int(1e6)], symbol='XBTUSD'),
               VWAP_volume(volumes=[int(1e5), int(5e5)], symbol='ETHUSD')]
    simulation = Strategy(metrics, [Filters.DepthFilter(3)])
    backtester = backtest.Backtest(reader, simulation, None)

    backtester.run()
    self.assertTrue(len(backtester.memory['XBTUSD']) > 2)

    # self.assertTrue(len(backtester.metrics["('XBTUSD', 'VWAP (Depth): 3 bid')"]) > 2)

  def test_init_moment(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz')
    callables = [('trades count', lambda trades: len(trades))]
    simulation = Strategy([], [], time_metrics=[TimeMetric(callables, 60)])
    backtester = backtest.Backtest(reader, simulation)

    row = reader.__next__()

    self.assertEqual((row.timestamp - simulation.time_metrics[0]._from).seconds, 0)



  def test_trades_len_minute_metric(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=1000)
    callables = [('trades count', lambda trades: len(trades))]
    simulation = Strategy([], [], time_metrics=[TimeMetric(callables, 60)])

    output = TestOutput([], simulation.time_metrics[0].metric_names)
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.time_metrics), 87)

  def test_trades_volume_minute_metric(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=10000, pairs_to_load=10)
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    simulation = Strategy([], [], time_metrics=[TimeMetric(callables, 60)])
    instant_metric_names = []

    output = TestOutput(instant_metric_names, simulation.time_metrics[0].metric_names)

    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.time_metrics), 2020)


  def test_all_metrics(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=10000, pairs_to_load=5)
    # todo: optimize return metrics (do not waste time on wrapping each -> transform into tuple of values with one header
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]

    metrics = [VWAP_volume(volumes=[100000, 1000000], symbol='XBTUSD'),
               VWAP_volume(volumes=[50000, 500000], symbol='ETHUSD')]
    simulation = Strategy(metrics, [Filters.DepthFilter(3)], time_metrics=[TimeMetric(callables, 60)])
    instant_metric_names = [metric.names() for metric in metrics]
    output = TestOutput(instant_metric_names, simulation.time_metrics[0].metric_names)
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.instant_metrics), 46336)

  def test_load_orderbooks_and_floats(self):
    reader = readers.OrderbookReader('resources/orderbook10/orderbook.csv.gz',
                                     'resources/orderbook10/trades.csv.gz',
                                     pairs_to_load=5)

    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]

    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    time_metrics = [TimeMetric(callables, 60), TimeMetric(callables, 30)]

    simulation = Strategy(instant_metrics, time_metrics=time_metrics)
    output = TestOutput(instant_metric_names=instant_metric_names,
                        time_metric_names=[metric.metric_names for metric in time_metrics])
    backtester = backtest.Backtest(reader, simulation, output)

    backtester.run()

    self.assertEqual(True, True)
