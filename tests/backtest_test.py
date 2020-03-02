import unittest

from backtesting import readers, backtest
from backtesting.output import TestOutput
from backtesting.trade_simulation import Simulation

from metrics.filters import Filters
from metrics.metrics import *


class BacktestTest(unittest.TestCase):
  def test_run(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz', stop_after=100)
    metrics = [VWAP_depth(3),
               VWAP_volume(volume=int(1e6), symbol='XBTUSD'),
               VWAP_volume(volume=int(1e5), symbol='ETHUSD')]
    simulation = Simulation(metrics, [Filters.DepthFilter(3)])
    backtester = backtest.Backtest(reader, simulation, None)

    backtester.run()
    self.assertTrue(len(backtester.memory['XBTUSD']) > 2)

    # self.assertTrue(len(backtester.metrics["('XBTUSD', 'VWAP (Depth): 3 bid')"]) > 2)

  def test_init_moment(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz')
    callables = [('trades count', lambda trades: len(trades))]
    simulation = Simulation([], [], time_metrics=[TimeMetric(callables, 60)])
    backtester = backtest.Backtest(reader, simulation)

    row = reader.__next__()

    self.assertEqual((row.timestamp - simulation.time_metrics[0]._from).seconds, 0)



  def test_trades_len_minute_metric(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=1000)
    callables = [('trades count', lambda trades: len(trades))]
    simulation = Simulation([], [], time_metrics=[TimeMetric(callables, 60)])
    output = TestOutput()
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.metrics), 87)

  def test_trades_volume_minute_metric(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=10000, depth=10)
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    simulation = Simulation([], [], time_metrics=[TimeMetric(callables, 60)])
    output = TestOutput()
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.metrics), 2020)


  def test_all_metrics(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=10000, depth=10)
    # todo: optimize return metrics (do not waste time on wrapping each -> transform into tuple of values with one header
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]

    metrics = [VWAP_depth(3),
               VWAP_volume(volume=int(1e6), symbol='XBTUSD'),
               VWAP_volume(volume=int(1e5), symbol='ETHUSD')]
    simulation = Simulation(metrics, [Filters.DepthFilter(3)], time_metrics=[TimeMetric(callables, 60)])
    output = TestOutput()
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.metrics), 46336)


