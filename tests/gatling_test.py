import unittest

from hft.backtesting import backtest
from hft.backtesting.output import StorageOutput
from hft.backtesting.readers import OrderbookReader
from hft.strategies.gatling import GatlingMM


# logging.disable(logging.CRITICAL)

class GatlingTest(unittest.TestCase):
  @unittest.skip("Skip as it is manual run")
  def test_gatling(self):
    reader = OrderbookReader('resources/huge_dataset/orderbook_10_03_20.csv.gz',
                             'resources/huge_dataset/trades_10_03_20.csv.gz',
                             stop_after=10000, depth_to_load=10, nrows=100000)

    output = StorageOutput([], [])
    output.balances = []
    strategy = GatlingMM(10000)
    strategy.balance_listener = output.balances.append

    backtester = backtest.Backtest(reader, strategy, delay=300)
    backtester.run()
