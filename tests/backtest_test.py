import unittest

from backtesting import readers, backtest
from backtesting.trade_simulation import Simulation


# class TestSimulation(Simulation):
#   pass
from metrics.filters import Filters
from metrics.metrics import *


class BacktestTest(unittest.TestCase):
  def test_rung(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz', stop_after=100)
    # todo: VWAP_volume PER SYMBOL !!!!!!!!!!!!
    simulation = Simulation([VWAP_depth(3), VWAP_volume()], [Filters.DepthFilter(3)])
    backtester = backtest.Backtest(reader, simulation, None)

    backtester.run()
    self.assertTrue(len(backtester.memory['XBTUSD']) > 2)

    # self.assertTrue(len(backtester.metrics["('XBTUSD', 'VWAP (Depth): 3 bid')"]) > 2)
