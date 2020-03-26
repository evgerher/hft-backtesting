import unittest

from backtesting import backtest
from backtesting.output import StorageOutput
from backtesting.readers import OrderbookReader
from strategies.gatling import GatlingMM
import logging
logging.disable(logging.CRITICAL)

class GatlingTest(unittest.TestCase):
  # @unittest.skip("Skip as it is manual run")
  def test_gatling(self):
    reader = OrderbookReader('resources/huge_dataset/orderbook_10_03_20.csv.gz',
                             'resources/huge_dataset/trades_10_03_20.csv.gz',
                             stop_after=300000, depth_to_load=8)

    output = StorageOutput([], [])
    output.balances = []
    strategy = GatlingMM()
    strategy.balance_listener = lambda b, t: output.balances.append((t, b))

    backtester = backtest.Backtest(reader, strategy, output=output, delay=300)
    backtester.run()
