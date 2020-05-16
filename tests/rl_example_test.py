import datetime
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
import glob

tick_counter = 0


class ModelTest(unittest.TestCase):
  # @unittest.skip('')
  def test_run(self):
    # matrices: [2,3,2], [2,3,2], [2, 2], [2]
    names = ['vwap', 'liquidity-spectrum', 'hayashi-yoshido', 'lipton']
    # [2, 2], [2, 2]
    time_names = ['trade-metric-45', 'trade-metric-75']

    class DuelingDQN(nn.Module):
      def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

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
        return value, adv, qvals

    State = namedtuple('State', 'prev_obs prev_ps action obs ps meta done')

    class DecisionCondition:
      def __init__(self, volume: float):
        self.volume_condition: float = volume
        self.volume = 0.0

      def __call__(self, event: Union[Trade, OrderBook]):
        if isinstance(event, Trade) and event.symbol == 'XBTUSD':
          self.volume += event.volume
          if self.volume > self.volume_condition:
            self.volume %= self.volume_condition
            return True
        return False

    class Agent:
      def __init__(self, model: DuelingDQN, target: DuelingDQN,
                   condition: DecisionCondition,
                   gamma = 0.99, lr=1e-3,
                   update_each:int=4,
                   buffer_size:int=30000, batch_size=1024):
        self._replay_buffer: Deque[State] = deque(maxlen=buffer_size)
        self._batch_size: int = batch_size
        self._condition: DecisionCondition = condition
        self._model : DuelingDQN = model
        self._target: DuelingDQN = target
        self.end_episode_states: List = []

        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 400


        self.gamma = gamma
        self.MSE_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self._model.parameters(), lr=lr)

        self._update_each = update_each
        self.episode_counter = 0
        self.reset_state()

      def reset_state(self):
        self.obs = None
        self.action = None
        self.ps = None
        self.episode_counter += 1

        # reload weights
        if (self.episode_counter + 1) % self._update_each == 0:
          self._target.load_state_dict(self._model.state_dict())

      def get_reward(self, prev_v: torch.Tensor, v: torch.Tensor,
                     prev_ps: torch.Tensor, ps: torch.Tensor,
                     tau: torch.Tensor, a=0.7, b=1./1000):
        vv = a * (v - prev_v)
        vv.squeeze_(1)

        state_delta = torch.abs(ps) - torch.abs(prev_ps)
        pos = (torch.exp(b*tau) * torch.sign(state_delta))

        # a(V_t - V_{t-1}) + e^{b*tau} * sgn(|i_t| - |i_{t-1}|)
        return vv + pos

      def get_terminal_reward(self, terminal_ps, terminal_prices, alpha=3.0):
        return alpha - torch.exp(-(terminal_ps[:, 0] / terminal_prices - terminal_ps[:, 1])) # 0: usd, 1: xbtusd, 2: ethusd

      def store_episode(self, new_obs, new_ps, meta, done, action):
        if self.is_initialized():
          self._replay_buffer.append((self.obs, self.ps, self.action, new_obs, new_ps, meta, done))
        self.obs = new_obs
        self.ps = new_ps
        self.action = action

      def is_initialized(self):
        return self.obs is not None  # happens after reset, first obs is missing

      def get_action(self, obs, ps, eps=0.8): # todo: epsilon must change accordging to curve
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * self.episode_counter / self.EPS_DECAY)
        if np.random.uniform(0.0, 1.0) < eps:
          return random.randint(0, self._model.output_dim - 1)
        with torch.no_grad():
          _, _, qvals = self._target(torch.tensor(obs, dtype=torch.float).unsqueeze(0))
          action = torch.argmax(qvals).cpu().detach().item()
          return action

      def update(self):
        if len(self._replay_buffer) > self._batch_size:
          items = random.sample(self._replay_buffer, self._batch_size)
          prev_obs, prev_ps, action, obs, ps, meta, done = zip(*items)
          prev_obs, prev_ps, obs, ps, meta = map(lambda x: torch.tensor(x, dtype=torch.float), [prev_obs, prev_ps, obs, ps, meta])
          done = torch.tensor(done, dtype=torch.bool)
          action = torch.tensor(action, dtype=torch.long).unsqueeze(1)

          v, _, qvalues = self._model(prev_obs) # todo: here are target and current models?
          next_v, _, next_qvalues = self._target(obs)

          rewards = torch.empty_like(done, dtype=torch.float)
          rewards[done] = self.get_terminal_reward(ps[done], meta[done][:, 0])
          rewards[~done] = self.get_reward(v[~done], next_v[~done], prev_ps[~done][:, 1], ps[~done][:, 1], meta[~done][:, 1])

          qvalues = qvalues.gather(1, action).squeeze(1)
          next_qvalues = next_qvalues.max(1)[0]
          expected_q = rewards + self.gamma * next_qvalues

          loss = self.MSE_loss(qvalues, expected_q)  # or hubert loss

          self.optimizer.zero_grad()
          loss.backward()
          for param in self._model.parameters():
            param.grad.data.clamp_(-1., 1.)
          self.optimizer.step()

    class RLStrategy(Strategy):
      def __init__(self, agent: Agent, simulation_end: datetime.datetime, **kwags):
        super().__init__(**kwags)
        self.agent: Agent = agent
        self._simulation_end = simulation_end
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

      def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
        obs, ps, meta = self.get_observation(), self.get_state(), (self.get_prices(memory), 0.0)
        self.agent.store_episode(obs, ps, meta, True, None)
        self.agent.end_episode_states.append(ps)
        # self.agent.update() # todo: remove this line later on
        self.agent.reset_state()

        super().return_unfinished(statuses, memory)

      def get_observation(self):
        # transform trivial metrics
        items = list(map(lambda name: self.metrics_map[name].to_numpy(), names))

        # transformation for time metrics
        # items += [np.array(list(map(lambda x: list(x.values()), self.metrics_map[name].latest.values()))) for name in time_names]
        items += list(map(lambda name: self.metrics_map[name].to_numpy(), time_names))
        items = [t.flatten() for t in items]
        items = np.concatenate(items, axis=None)
        if len(items) != 38:
          print('here')
        return items
        # return items

      def get_prices(self, memory) -> float: # todo: refactor and use vwap
        xbt: OrderBook = memory[('orderbook', 'XBTUSD')]
        # eth: OrderBook = memory[('orderbook', 'ETHUSD')]

        xbt_midprice = (xbt.ask_prices[0] + xbt.bid_prices[0]) / 2
        # eth_midprice = (eth.ask_prices[0] + eth.bid_prices[0]) / 2

        # return eth_midprice / xbt_midprice
        return xbt_midprice

      def get_state(self) -> np.array:
        return np.array(list(self.balance.values()))

      def get_timeleft(self, ts: datetime.datetime) -> float:
        return (self._simulation_end - ts).total_seconds()

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
        if self.agent._condition(row):
          obs = self.get_observation()
          ps = self.get_state()
          action = self.agent.get_action(obs, ps)
          meta = (self.get_prices(memory), self.get_timeleft(row.timestamp))

          self.agent.store_episode(obs, ps, meta, False, action)
          self.agent.update()

          orders = self.action_to_order(action, memory, row.timestamp, 1000)
        else:
          orders = []
        return orders

    def init_simulation(agent: Agent, orderbook_file: str, trade_file: str):
      vwap = VWAP_volume([int(2.5e5), int(1e6)], name='vwap', z_normalize=4000)
      liq = LiquiditySpectrum(z_normalize=4000)

      trade_metric = TradeMetric([
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: np.log(sum(map(lambda x: x.volume, trades))))
      ], seconds=45) # todo: add z-normalize for time-metrics
      trade_metric2 = TradeMetric([
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: np.log(sum(map(lambda x: x.volume, trades))))
      ], seconds=75)

      hy = HayashiYoshido(seconds=75)
      lipton = Lipton(hy.name)

      # todo: refactor in backtesting auto-cancel queries with prices worse than top 3 levels
      # todo: update reader to work with only `xbtusd`
      reader = OrderbookReader(orderbook_file, trade_file, nrows=None, is_precomputed=True)
      end_ts = reader.get_ending_moment()

      strategy = RLStrategy(agent, simulation_end=end_ts, instant_metrics=[vwap, liq], delta_metrics=[hy],
                            time_metrics_trade=[trade_metric, trade_metric2], composite_metrics=[lipton], initial_balance=0.0)
      backtest = BacktestOnSample(reader, strategy, delay=300, warmup=True, stale_depth=5)
      backtest.run()

    condition = DecisionCondition(150000.0)
    model: DuelingDQN = DuelingDQN(input_dim=38, output_dim=9)
    target: DuelingDQN = DuelingDQN(input_dim=38, output_dim=9)
    target.load_state_dict(model.state_dict())

    agent = Agent(model, target, condition, batch_size=512)
    for idx, (ob_file, tr_file) in enumerate(zip(glob.glob('../notebooks/time-sampled/orderbook_*'), glob.glob('../notebooks/time-sampled/trade_*'))):
      init_simulation(agent, ob_file, tr_file)
      if idx == 3:
        break

    end_states = agent.end_episode_states
    for t in end_states:
      print(t)
