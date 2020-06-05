import datetime
import unittest
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

from hft.backtesting.data import OrderRequest
from hft.utils.data import OrderBook, Trade

from hft.backtesting import backtest
from hft.backtesting.output import StorageOutput, make_plot_orderbook_trade, SimulatedOrdersOutput
from hft.backtesting.readers import OrderbookReader, TimeLimitedReader
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

  # @unittest.skip("Skip as it is manual run")
  def test_timelimited_gatling(self):
    class OrdersOutput(StorageOutput):
      def __init__(self, instant_metric_names, time_metric_names):
        super().__init__(instant_metric_names, time_metric_names)
        self.prices = defaultdict(list)
        self.orders = defaultdict(list)
        self.snapshots = defaultdict(list)
        self.trades = defaultdict(list)

      def time_metric_action(self, timestamp, labels, object):
        pass

      def snapshot_action(self, timestamp: datetime.datetime, object: OrderBook):
        self.snapshots[object.symbol].append((timestamp, object.bid_prices[0], object.ask_prices[0]))

      def instant_metric_action(self, timestamp, labels, object):
        pass

      def trade_action(self, timestamp: datetime.datetime, object: Trade):
        self.trades[(object.symbol, object.side)].append((object.timestamp, object.price, object.volume))

      def additional_action(self, timestamp: datetime.datetime, labels, object: OrderRequest):
        if 'order-request' in labels:
          self.orders[(labels[1], labels[2])].append((timestamp, object.price, object.volume))

      def metric_action(self, timestamp: datetime.datetime, object):
        pass

    reader = TimeLimitedReader('resources/huge_dataset/orderbook_10_03_20.csv.gz',
                             limit_time='100 min',
                             skip_time=None,
                             trades_file='resources/huge_dataset/trades_10_03_20.csv.gz',
                             nrows=500000)

    # output = OrdersOutput([], [])
    # output.balances = []
    output = SimulatedOrdersOutput()
    output.balances = []

    strategy = GatlingMM(1000, initial_balance=0.0)
    strategy.balance_listener = output.balances.append

    backtester = backtest.Backtest(reader, strategy, output, stale_depth=1, delay=1)
    backtester.run()

    ord1, ord2 = list(output.orders.values())
    make_plot_orderbook_trade('resources/huge_dataset/orderbook_10_03_20.csv.gz', 'XBTUSD',
                              simulated_orders=ord1+ord2, nrows=1000000)
    plt.show()

    states = output.balances
    states = pd.DataFrame(states, index=None, columns=None)[[0, 1, 2, 3]].loc[1:]
    states.columns = ['usd', 'xbt', 'xbt_price', 'ts']
    states['nav'] = states.usd + states.xbt * states.xbt_price

    states.ts = pd.to_datetime(states.ts)
    plt.plot(states.ts, states.nav)
    plt.show()
    print(states.nav.loc[len(states) - 3:])

  @unittest.skip('custom')
  def test_me(self):
    make_plot_orderbook_trade('resources/huge_dataset/orderbook_10_03_20.csv.gz', 'XBTUSD', nrows=1000000)
    plt.show()

if __name__ == '__main__':
  unittest.main()
