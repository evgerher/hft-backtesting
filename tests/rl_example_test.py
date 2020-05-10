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
    # matrices: [2,3,2], [2,3,2], [2, 2], [2]
    names = ['vwap', 'liquidity-spectrum', 'hayashi-yoshido', 'lipton']
    # [2, 4]
    time_names = ['trade-metric-60']



    class RLStrategy(Strategy):

      def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
        obs = self.get_observation()
        portfolio_state = self.get_state()
        # reward = self.get_reward(state) + self.terminal_reward(portfolio_state)
        # todo: store terminal state and terminal reward
        super().return_unfinished(statuses, memory)

      def get_observation(self):

        # transform trivial metrics
        items = list(map(lambda name: np.array(list(self.metrics_map[name].latest.values()), dtype=np.float), names))

        # transformation for time metrics
        items += [np.array(list(map(lambda x: list(x.values()), self.metrics_map[name].latest.values()))) for name in time_names]
        return items


      def get_state(self):
        # `position` is expected to be initialized
        self.position['XBTUSD'] = (7846., 0.003)  # just for demonstration purpose
        self.position['ETHUSD'] = (143.9, -0.04)  # just for demonstration purpose
        return np.array(list(self.position.values()))[:, 1] # [2, ]

      def reward(self):
        pass

      def get_action(self, *args):
        pass

      def define_orders(self, row: Union[Trade, OrderBook],
                        statuses: List[OrderStatus],
                        memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
        global tick_counter
        if (tick_counter + 1) % 1500 == 0:
          obs = self.get_observation()
          portfolio_state = self.get_state()
          # reward = self.get_reward(state)
          # action = self.get_action(obs, state)

          # todo: define risk-adjusted reward, model arch, storaging of SARSA?, weights update
          # todo: define initial obs_state pair from environment (first access to .define_orders)

          # todo: store state, reward and define action

          orders = []
        else:
          orders = []

        tick_counter += 1
        return orders

    def init_simulation(orderbook_file, trade_file):

      vwap1 = VWAP_volume([int(2.5e5), int(1e6)], name='vwap', z_normalize=3000)
      liq = LiquiditySpectrum(z_normalize=3000)

      trade_metric = TradeMetric([
        ('quantity', lambda x: len(x)),
        ('total', lambda trades: sum(map(lambda x: x.volume, trades)))
      ], seconds=60)

      hy = HayashiYoshido()
      lipton = Lipton(hy.name)

      reader = OrderbookReader(orderbook_file, trade_file, nrows=None, is_precomputed=True)
      strategy = RLStrategy([vwap1, liq], [hy], [trade_metric], composite_metrics=[lipton], initial_balance=0.0)
      backtest = BacktestOnSample(reader, strategy, delay=300, warmup=True)
      backtest.run()

    init_simulation('../notebooks/time-sampled/orderbook_0.csv.gz', '../notebooks/time-sampled/trade_0.csv.gz')
    print(tick_counter)
