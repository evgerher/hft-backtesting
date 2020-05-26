import datetime
import random
import unittest
from collections import deque, namedtuple, defaultdict
from typing import Union, List, Dict, Tuple, Deque, Optional
import pandas as pd

from hft.backtesting.backtest import BacktestOnSample
from hft.backtesting.data import OrderStatus, OrderRequest
from hft.backtesting.output import make_plot_orderbook_trade, Output, SimulatedOrdersOutput
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
    ################    DDQN    ################
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
      def __init__(self, mu: float, std: float = 1.0):
        assert mu > 0
        assert std > 0

        self.mu = mu
        self.std = std
        self.volume_condition: float = random.gauss(mu, std)
        self.reset()

      def __call__(self, is_trade: bool, event: Union[Trade, OrderBook]):
        if is_trade and event.symbol == 'XBTUSD':
          self.volume += event.volume
          if self.volume > self.volume_condition:
            self.volume %= self.volume_condition
            self.volume_condition = random.gauss(self.mu, self.std)
            return True
        return False

      def reset(self):
        self.volume = 0.0

    ################    RL agent operations wrapper    ################
    class Agent:
      def __init__(self, model: DuelingDQN, target: DuelingDQN,
                   condition: DecisionCondition,
                   gamma=0.95, lr=1e-3,
                   update_each: int = 3,
                   buffer_size: int = 30000, batch_size=1024):
        self._replay_buffer: Deque[State] = deque(maxlen=buffer_size)
        self._batch_size: int = batch_size
        self.condition: DecisionCondition = condition
        self._model: DuelingDQN = model
        self._target: DuelingDQN = target
        self.end_episode_states: List = []
        self.episode_files: List[Tuple[str, str]] = []

        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 500

        self.gamma = gamma
        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, amsgrad=True)

        self._update_each = update_each
        self.episode_counter = 0

        self.no_action_event = deque(maxlen=20)
        self.reset_state()

      def store_no_action(self, ts, price):
        self.no_action_event[0].append((ts, price))

      def episode_results(self) -> pd.DataFrame:
        fnames, states = agent.episode_files, agent.end_episode_states
        states = [t[0].tolist() + [t[1]] for t in states]
        episodes = [list(t[0]) + t[1] for t in zip(fnames, states)]
        res = pd.DataFrame(episodes, columns=['ob_file', 'tr_file', 'usd', 'xbt', 'eth', 'xbt_price'])
        return res

      def reset_state(self):
        self.obs = None
        self.action = None
        self.ps = None
        self.episode_counter += 1
        self.condition.reset()
        self.no_action_event.appendleft([])

        # reload weights
        state = self._model.state_dict()
        if (self.episode_counter + 1) % self._update_each == 0:
          self._target.load_state_dict(state)
          torch.save(state, f'models/model-{self.episode_counter}.pth')
        torch.save(state, f'model-latest.pth')

      def get_reward(self, prev_v: torch.Tensor, v: torch.Tensor,
                     prev_ps: torch.Tensor, ps: torch.Tensor,
                     tau: torch.Tensor, a=0.1, b=1. / 1000,
                     offset=1.1, scale=0.1):
        vv = a * (v - prev_v).squeeze(1)

        state_delta = torch.abs(prev_ps) - torch.abs(
          ps)  # todo: updated here, react negatively on accumulation of assets
        pos = (torch.exp(b * tau) * state_delta)  # todo: updated here, instead of sign, use delta

        accumulation_penalty = -torch.exp(torch.abs(ps) * scale) + offset

        # a(V_t - V_{t-1}) + e^{b*tau} * sgn(|i_t| - |i_{t-1}|)
        return vv + pos + accumulation_penalty

      def get_terminal_reward(self, terminal_ps, terminal_prices, alpha=3.0, scale=4.0, shift=8.0):
        inbalance = (terminal_ps[:, 0] / terminal_prices - terminal_ps[:, 1])
        return torch.exp(torch.abs(inbalance) * scale) * torch.sign(inbalance) / shift  # 0: usd, 1: xbtusd, 2: ethusd

      def store_episode(self, new_obs, new_ps, meta, done, action):
        nans = [np.isnan(t).any() for t in [new_obs, new_ps, meta, done, action]]
        if not any(nans):
          if self.is_initialized():
            self._replay_buffer.append((self.obs, self.ps, self.action, new_obs, new_ps, meta, done))
          self.obs = new_obs
          self.ps = new_ps
          self.action = action

      def is_initialized(self):
        return self.obs is not None  # happens after reset, first obs is missing

      def get_action(self, obs):
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * self.episode_counter / self.EPS_DECAY)
        if np.random.uniform(0.0, 1.0) < eps:
          return random.randint(0, self._model.output_dim - 1)
        with torch.no_grad():
          _, _, qvals = self._target(torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0))
          action = torch.argmax(qvals).cpu().detach().item()
          return action

      def update(self):
        if len(self._replay_buffer) > self._batch_size:
          items = random.sample(self._replay_buffer, self._batch_size)
          prev_obs, prev_ps, action, obs, ps, meta, done = zip(*items)
          prev_obs, prev_ps, obs, ps, meta = map(lambda x: torch.tensor(x, dtype=torch.float).to(device),
                                                 [prev_obs, prev_ps, obs, ps, meta])
          done = torch.tensor(done, dtype=torch.bool).to(device)
          action = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(device)

          v, _, qvalues = self._model(prev_obs)
          with torch.no_grad():
            next_v, _, next_qvalues = self._target(obs[~done])

            #           next_v.detach()
            #           next_qvalues.detach()
            #           _.detach()
            next_qvalues = next_qvalues.max(1)[0]

          rewards = torch.empty_like(done, dtype=torch.float, device=device)
          rewards[done] = self.get_terminal_reward(ps[done], meta[done][:, 0])
          rewards[~done] = self.get_reward(v[~done], next_v, prev_ps[~done][:, 1], ps[~done][:, 1], meta[~done][:, 1])
          rewards = rewards.clamp(-1., 1.)  # reward clipping

          qvalues = qvalues.gather(1, action).squeeze(1)

          expected_q = torch.empty_like(rewards, dtype=torch.float, device=device)
          expected_q[~done] = self.gamma * next_qvalues
          expected_q += rewards

          loss = self.loss(qvalues, expected_q)  # or hubert loss

          self.optimizer.zero_grad()
          loss.backward()
          for param in self._model.parameters():
            param.grad.data.clamp_(-1., 1.)  # gradient clipping
          self.optimizer.step()

    ################    Strategy wrapper    ################
    class RLStrategy(Strategy):
      def __init__(self, agent: Agent, simulation_end: datetime.datetime, cancels_enabled=False, **kwags):
        super().__init__(**kwags)
        self.agent: Agent = agent
        self._simulation_end = simulation_end
        self.action_space: Dict[int, Tuple[int, int]] = {
          0: (0, 0),
          1: (1, 0),
          2: (2, 0),
          3: (3, 0),
          4: (0, 1),
          5: (1, 1),
          6: (1, 2),
          7: (1, 3),
          8: (0, 3),
          9: (2, 1),
          10: (2, 2),
          11: (2, 3),
          12: (0, 2),
          13: (3, 1),
          14: (3, 2),
          15: (3, 3),
          16: (4, 0),
          17: (4, 1),
          18: (4, 2),
          19: (4, 3),
          20: (4, 4),
          21: (0, 4),
          22: (1, 4),
          23: (2, 4),
          24: (3, 4),
        }
        self.cancels_enabled = cancels_enabled
        self.no_action_event = []

      def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
        obs, ps, prices = self.get_observation(memory, 0.0)
        meta = (prices, 0.0)
        self.agent.store_episode(obs, ps, meta, True, 0)
        self.agent.end_episode_states.append((ps, meta[0]))  # end state and prices
        self.agent.reset_state()

        super().return_unfinished(statuses, memory)

      def get_observation(self, memory, timeleft):
        items = list(map(lambda name: self.metrics_map[name].to_numpy(), names + time_names))
        items = [t.flatten() for t in items]

        prices = self.get_prices(memory)
        ps = self.get_state()
        state = np.array([ps[0] / prices, ps[1]])

        items += [state, timeleft]
        items = np.concatenate(items, axis=None)
        return items, ps, prices

      def get_prices(self, memory) -> float:  # todo: refactor and use vwap
        xbt: OrderBook = memory[('orderbook', 'XBTUSD')]
        xbt_midprice = (xbt.ask_prices[0] + xbt.bid_prices[0]) / 2

        # eth: OrderBook = memory[('orderbook', 'ETHUSD')]
        # eth_midprice = (eth.ask_prices[0] + eth.bid_prices[0]) / 2

        return xbt_midprice

      def get_state(self) -> np.array:
        return np.array(list(self.balance.values()))

      def get_timeleft(self, ts: datetime.datetime) -> float:
        return (self._simulation_end - ts).total_seconds()

      def action_to_order(self, action: int, memory, ts, quantity: int) -> List[OrderRequest]:
        offset_bid, offset_ask = self.action_space[action]

        offset_bid *= 0.5  # price step is .5 dollars
        offset_ask *= 0.5

        ob: OrderBook = memory[('orderbook', 'XBTUSD')]
        orders = []

        if offset_bid > 0:
          orders.append(OrderRequest.create_bid(ob.bid_prices[0] - offset_bid, quantity, 'XBTUSD', ts))
        if offset_ask > 0:
          orders.append(OrderRequest.create_ask(ob.ask_prices[0] + offset_ask, quantity, 'XBTUSD', ts))

        return orders

      def cancel_orders(self, statuses: List[OrderStatus]) -> List[OrderRequest]:
        statuses_ids = list(map(lambda x: x.id, statuses))
        active = self.active_orders.items()
        active_non_present = filter(lambda x: x[0] not in statuses_ids, active)
        return list(map(lambda x: OrderRequest.cancelOrder(x[0]), active_non_present))

      def define_orders(self, row: Union[Trade, OrderBook],
                        statuses: List[OrderStatus],
                        memory: Dict[str, Union[Trade, OrderBook]],
                        is_trade: bool) -> List[OrderRequest]:
        if self.agent.condition(is_trade, row):
          orders = []
          timeleft = self.get_timeleft(row.timestamp)
          obs, ps, prices = self.get_observation(memory, timeleft)
          if self.cancels_enabled:
            cancels = self.cancel_orders(statuses)
            orders += cancels

          action = self.agent.get_action(obs)
          meta = (prices, timeleft)
          
          self.agent.store_episode(obs, ps, meta, False, action)
          self.agent.update()

          orders += self.action_to_order(action, memory, row.timestamp, 1000)

        else:
          orders = []
        return orders

    ################    Simulation wrapper    ################
    def init_simulation(agent: Agent, orderbook_file: str, trade_file: str,
                        output_required: Union[bool, Output] = False, delay=5,
                        cancels_enaled=True) -> Optional[Output]:

      if isinstance(output_required, bool) and output_required:
        output = SimulatedOrdersOutput()
      elif isinstance(output_required, Output):
        output = output_required
      else:
        output = None

      vwap = VWAP_volume([int(5e5), int(1e6), int(2e6)], name='vwap', z_normalize=3000)
      liq = LiquiditySpectrum(z_normalize=3000)

      defaults = [
        (('XBTUSD', 0), [0.0]),
        (('XBTUSD', 1), [0.0]),
        (('ETHUSD', 0), [0.0]),
        (('ETHUSD', 1), [0.0]),
      ]

      trade_metric = TradeMetric(defaults, [
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: sum(map(lambda x: x.volume, trades)))
      ], seconds=45, z_normalize=2000)
      trade_metric2 = TradeMetric(defaults, [
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: sum(map(lambda x: x.volume, trades)))
      ], seconds=80, z_normalize=2000)

      lipton_levels = 8
      hy = HayashiYoshido(140, True)
      lipton = Lipton(hy.name, lipton_levels)

      reader = OrderbookReader(orderbook_file, trade_file, nrows=None, is_precomputed=True)
      end_ts = reader.get_ending_moment()

      strategy = RLStrategy(agent, simulation_end=end_ts, instant_metrics=[vwap, liq], delta_metrics=[hy],
                            time_metrics_trade=[trade_metric, trade_metric2], composite_metrics=[lipton],
                            initial_balance=0.0, cancels_enabled=cancels_enaled)
      backtest = BacktestOnSample(reader, strategy, output=output, delay=delay, warmup=True, stale_depth=8)
      backtest.run()

      return backtest.output

    # matrices: [3,3,2], [2,3,2], [2, 2], [2]
    names = ['vwap', 'liquidity-spectrum', 'hayashi-yoshido', 'lipton']
    # [2, 2], [2, 2]
    time_names = ['trade-metric-45', 'trade-metric-80']

    ### Initialize agent, model, target network, decision unit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    condition = DecisionCondition(100000.0)
    model: DuelingDQN = DuelingDQN(input_dim=47, output_dim=25)
    target: DuelingDQN = DuelingDQN(input_dim=47, output_dim=25)
    # model.load_state_dict(torch.load('model-latest.pth'))
    model.train()
    target.eval()
    target.load_state_dict(model.state_dict())

    # target.load_state_dict(torch.load('notebooks/model.pth'))
    model.to(device)
    target.to(device)
    agent = Agent(model, target, condition, batch_size=8)

    ### Initialize simulation
    pairs = list(zip(glob.glob('../notebooks/time-sampled/orderbook_*'), glob.glob('../notebooks/time-sampled/trade_*')))
    ob_file, tr_file = random.choice(pairs)
    result_output = init_simulation(agent, ob_file, tr_file, output_required=True, cancels_enaled=True)
    agent.episode_files.append((ob_file, tr_file))

    ### Visualize results
    orders_side1, orders_side2 = list(result_output.orders.values())
    make_plot_orderbook_trade(ob_file, 'XBTUSD',
                              orders_side1 + orders_side2,
                              orderbook_precomputed=True)

    res = agent.episode_results()
    res['pnl'] = res['usd'] + res['xbt'] * res['xbt_price']
    print(res)




  def test_plot(self):
    make_plot_orderbook_trade('../notebooks/time-sampled-10min/orderbook_1715.csv.gz', 'XBTUSD', orderbook_precomputed=True)
    print('ok')
