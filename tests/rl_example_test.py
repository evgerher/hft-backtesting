import random
import unittest
from collections import deque, namedtuple
from typing import Union, List, Dict, Tuple, NamedTuple, Deque

from hft.backtesting.backtest import BacktestOnSample
from hft.backtesting.data import OrderStatus, OrderRequest
from hft.backtesting.readers import OrderbookReader
from hft.backtesting.strategy import Strategy
from hft.units.metrics.composite import Lipton
from hft.units.metrics.instant import VWAP_volume, LiquiditySpectrum, HayashiYoshido
from hft.units.metrics.time import TradeMetric
from hft.utils.data import Trade, OrderBook
import numpy as np
import torch
from torch import nn


tick_counter = 0


class ModelTest(unittest.TestCase):
  # @unittest.skip('')
  def test_run(self):
    # matrices: [2,3,2], [2,3,2], [2, 2], [2]
    names = ['vwap', 'liquidity-spectrum', 'hayashi-yoshido', 'lipton']
    # [2, 4]
    time_names = ['trade-metric-60', 'trade-metric-120']


    class DuelingDQN(nn.Module):
      def __init__(self, input_dim, output_dim, gamma=0.99, lr=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        self.feature_layer = nn.Sequential(
          nn.Linear(input_dim, 128),
          nn.LayerNorm(128),
          nn.ReLU(),
          nn.Linear(128, 128),
          nn.LayerNorm(128),
          nn.ReLU()
        )

        self.value_stream = nn.Sequential(
          nn.Linear(128, 128),
          nn.LayerNorm(128),
          nn.ReLU(),
          nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
          nn.Linear(128, 128),
          nn.LayerNorm(128),
          nn.ReLU(),
          nn.Linear(128, self.output_dim)
        )

      def forward(self, x):
        x = self.feature_layer(x)
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        qvals = value + (adv - adv.mean())
        return qvals

    State = namedtuple('State', 'prev_obs prev_ps action obs ps prices done' )


    class RLStrategy(Strategy):
      def __init__(self, model: DuelingDQN, buffer_length: int, **kwags):
        super().__init__(**kwags)
        self.buffer: Deque[State] = deque(maxlen=buffer_length)
        self.model: DuelingDQN = model
        self.action_space: Dict[int, Tuple[int, int]] = {
          0: (0, 0),
          1: (1, 0),
          2: (2, 0),
          3: (0, 1),
          4: (1, 1),
          5: (1, 2),
          6: (2, 0),
          7: (2, 1),
          8: (2, 2)
        }
        # work only with xbtusd
        # self.position: Dict[str, Tuple[float, float]] = {'XBTUSD': (0.0, 0.0)}  # (average_price, volume)

      def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
        obs = self.get_observation()
        ps = self.get_state()
        prices = self.get_prices(memory)
        self.buffer.append((self.prev_obs, self.prev_ps, self.prev_action, obs, ps, prices, True))
        super().return_unfinished(statuses, memory)

      def get_observation(self):
        # transform trivial metrics
        items = list(map(lambda name: np.array(list(self.metrics_map[name].latest.values()), dtype=np.float), names))

        # transformation for time metrics
        items += [np.array(list(map(lambda x: list(x.values()), self.metrics_map[name].latest.values()))) for name in time_names]
        items = list(map(np.flatten, items))
        return np.concatenate([items])
        # return items

      def get_prices(self, memory) -> float: # todo: refactor and use vwap
        xbt: OrderBook = memory[('orderbook', 'XBTUSD')]
        # eth: OrderBook = memory[('orderbook', 'ETHUSD')]

        xbt_midprice = (xbt.ask_prices[0] + xbt.bid_prices[0]) / 2
        # eth_midprice = (eth.ask_prices[0] + eth.bid_prices[0]) / 2

        # return eth_midprice / xbt_midprice
        return xbt_midprice

      # def get_state(self) -> np.array:
      #   # `position` is expected to be initialized
      #   self.position['XBTUSD'] = (7846., 0.003)  # just for demonstration purpose
      #   self.position['ETHUSD'] = (143.9, -0.04)  # just for demonstration purpose
      #   return np.array(list(self.position.values()))[:, 1] # [2, ]

      def get_state(self) -> np.array:
        return np.array(list(self.balance.values()))
        # return np.array(list(self.position.values()))[:, 1]

      def get_reward(self, prev_v: torch.Tensor, v: torch.Tensor,
                     prev_ps: np.array, ps: np.array,
                     tau: float, a=0.7, b=0.01):
        vv = a * (v - prev_v)

        state_delta = np.abs(ps[1, :]) - np.abs(prev_ps[1, :])
        pos = np.exp(b*tau) * np.sign(state_delta)

        # a(V_t - V_{t-1}) + e^{b*tau} * sgn(|i_t| - |i_{t-1}|)
        return vv + pos

      def get_terminal_reward(self, terminal_ps, terminal_prices, alpha=3.0):
        return alpha - torch.exp(-(terminal_ps[0, :] - terminal_ps[1, :] * terminal_prices))

      def update(self):
        items = random.sample(self.buffer, 1024)
        prev_obs, prev_ps, action, obs, ps, prices, done = map(torch.tensor, zip(*items))

        prev_v, _, prev_qvalues = self.model(prev_obs[~done]) # todo: here are target and current models?
        v, _, qvalues = self.model(obs[~done])

        rewards = torch.empty_like(done)
        rewards[done] = self.get_terminal_reward(ps[done], prices[done])
        rewards[~done] = self.get_reward(prev_v, v, prev_ps, ps, tau, a, b)

        prev_qvalues = prev_qvalues.gather(1, action)
        qvalues = qvalues.max(1)[0]
        expected_q = rewards + self.model.gamma * qvalues

        loss = self.model.MSE_loss(prev_qvalues, expected_q) # or hubert loss

        self.model.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
          param.grad.data.clamp_(-1., 1.)
        self.model.optimizer.step()

      def get_action(self, obs, ps, eps=0.2): # todo: epsilon must change accordgin to curve
        if np.random.uniform(0.0, 1.0) < eps:
          return random.randint(0, len(self.action_space) - 1)
        with torch.no_grad():
          _, _, qvals = self.model(torch.tensor(obs).unsqueeze(0))
          action = torch.argmax(qvals).cpu().detach().item()
          return action

      def action_to_order(self, action: int, memory, ts, quantity: int) -> List[OrderRequest]: # value from 0 to 8
        offset_bid, offset_ask = self.action_space[action]

        offset_bid *= 0.5 # price step is .5 dollars
        offset_ask *= 0.5

        ob: OrderBook = memory[('orderbook', 'XBTUSD')]

        return [OrderRequest.create_bid(ob.bid_prices[0] - offset_bid, quantity, 'XBTUSD', ts),
                OrderRequest.create_ask(ob.ask_prices[0] + offset_ask, quantity, 'XBTUSD', ts)]

      def define_orders(self, row: Union[Trade, OrderBook],
                        statuses: List[OrderStatus],
                        memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
        global tick_counter
        if (tick_counter + 1) % 1500 == 0: # todo: refactor condition
          obs = self.get_observation()
          ps = self.get_state()
          action = self.get_action(obs, ps)
          prices = self.get_prices(memory)

          if self.prev_obs is not None:
            self.buffer.append((self.prev_obs, self.prev_ps, self.prev_action, obs, ps, prices, False))

          self.prev_ps = ps
          self.prev_obs = obs
          self.prev_action = action

          if len(self.buffer) > 1024:
            self.update()

          orders = self.action_to_order(action, memory, row.timestamp, 1000)
        else:
          orders = []

        tick_counter += 1
        return orders

    def init_simulation(orderbook_file, trade_file):

      vwap1 = VWAP_volume([int(2.5e5), int(1e6)], name='vwap', z_normalize=1500)
      liq = LiquiditySpectrum(z_normalize=1500)

      trade_metric = TradeMetric([
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: np.log(sum(map(lambda x: x.volume, trades))))
      ], seconds=60)
      trade_metric2 = TradeMetric([
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: np.log(sum(map(lambda x: x.volume, trades))))
      ], seconds=180)

      hy = HayashiYoshido()
      lipton = Lipton(hy.name)

      # todo: refactor in backtesting auto-cancel queries with prices worse than top 3 levels
      # todo: update reader to work with only `xbtusd`
      model = DuelingDQN(input_dim=19, output_dim=9)
      reader = OrderbookReader(orderbook_file, trade_file, nrows=None, is_precomputed=True)
      strategy = RLStrategy(model, buffer_length=40000, instant_merics=[vwap1, liq], delta_metrics=[hy],
                            time_metrics_trade=[trade_metric, trade_metric2], composite_metrics=[lipton], initial_balance=0.0)
      backtest = BacktestOnSample(reader, strategy, delay=300, warmup=True)
      backtest.run()

    init_simulation('../notebooks/time-sampled/orderbook_0.csv.gz', '../notebooks/time-sampled/trade_0.csv.gz')
    print(tick_counter)
