import unittest
from typing import Union, List, Dict

from hft.backtesting.backtest import BacktestOnSample
from hft.backtesting.data import OrderStatus, OrderRequest
from hft.backtesting.readers import OrderbookReader
from hft.backtesting.strategy import Strategy
from hft.units.metrics.composite import Lipton
from hft.units.metrics.instant import VWAP_volume, LiquiditySpectrum, HayashiYoshido
from hft.units.metrics.time import TradeMetric
from hft.utils.data import Trade, OrderBook
import numpy as np


tick_counter = 0


class ModelTest(unittest.TestCase):
  # @unittest.skip('')
  def test_run(self):
    names = ['vwap', 'liquidity-spectrum', 'trade-metric-60', 'hayashi-yoshido', 'lipton']

    class RLStrategy(Strategy):

      def get_state(self):
        # vwap1 = np.fromiter(self.metrics_map['vwap1'].latest.values(), dtype=np.float) # np array [2, 3, 2]
        # liq   = np.fromiter(self.metrics_map['liquidity-spectrum'].latest.values(), dtype=np.float) # np array [2, 3]
        # trades = np.fromiter(self.metrics_map['trade-metric-60'].latest.values(), dtype=np.float) # np array [2, 2] ???
        # hy = np.fromiter(self.metrics_map['hayashi-yoshido'].latest.values(), dtype=np.float) # np array [2, 2]
        # lipton = np.fromiter(self.metrics_map['lipton'].latest.values(), dtype=np.float) # np array
        # todo: may be without list() for dict_values collection?
        items = list(map(lambda name: np.array(list(self.metrics_map[name].latest.values()), dtype=np.float), names))
        return items

      def define_orders(self, row: Union[Trade, OrderBook],
                        statuses: List[OrderStatus],
                        memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
        global tick_counter
        if (tick_counter + 1) % 3000 == 0:
          obs = self.get_state()
          obs = np.stack(obs)

          orders = []
        else:
          orders = []

        tick_counter += 1
        return orders

    def init_simulation(orderbook_file, trade_file):

      vwap1 = VWAP_volume([int(2.5e5), int(1e6)], name='vwap')
      liq = LiquiditySpectrum()

      trade_metric = TradeMetric([
        ('quantity', lambda x: len(x)),
        ('total', lambda trades: sum(map(lambda x: x.volume, trades)))
      ], seconds=60)

      hy = HayashiYoshido()
      lipton = Lipton(hy.name)

      reader = OrderbookReader(orderbook_file, trade_file, nrows=None, is_precomputed=True)
      strategy = RLStrategy([vwap1, liq], [hy], [trade_metric], composite_metrics=[lipton], initial_balance=0.0)
      backtest = BacktestOnSample(reader, strategy, delay=300)
      backtest.run()

    init_simulation('../notebooks/time-sampled/orderbook_0.csv.gz', '../notebooks/time-sampled/trade_0.csv.gz')
    print(tick_counter)
