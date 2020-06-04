import datetime
import random
import unittest
from collections import deque, namedtuple, defaultdict
from typing import Union, List, Dict, Tuple, Deque, Optional
import pandas as pd
from hft.strategies.gatling import GatlingMM

from hft.backtesting.backtest import BacktestOnSample, Backtest
from hft.backtesting.data import OrderStatus, OrderRequest
from hft.backtesting.output import make_plot_orderbook_trade, Output, SimulatedOrdersOutput
from hft.backtesting.readers import OrderbookReader, TimeLimitedReader
from hft.backtesting.strategy import Strategy, CancelsApplied
from hft.units.metrics.composite import Lipton
from hft.units.metrics.instant import VWAP_volume, LiquiditySpectrum, HayashiYoshido
from hft.units.metrics.time import TradeMetric
from hft.utils.data import Trade, OrderBook
import numpy as np
import glob
import matplotlib.pyplot as plt

tick_counter = 0


class ModelTest(unittest.TestCase):
  # @unittest.skip('')
  def test_run(self):
    ################    DDQN    ################
    import torch
    from torch import nn

    class DuelingDQN(nn.Module):
      def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        self.feature_layer = nn.Sequential(
          nn.Linear(input_dim, 256),
          nn.LayerNorm(256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.LayerNorm(256),
          nn.ReLU()
        )

        self.value_stream = nn.Sequential(
          nn.Linear(256, 128),
          nn.LayerNorm(128),
          nn.ReLU(),
          nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
          nn.Linear(256, 256),
          nn.LayerNorm(256),
          nn.ReLU(),
          nn.Linear(256, self.output_dim)
        )

      def forward(self, x):
        x = self.feature_layer(x)
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        qvals = value + (adv - adv.mean())
        return value, adv, qvals

    State = namedtuple('State', 'prev_obs action obs done')

    class DecisionCondition:
      def __init__(self, mu: float, std: float = 1.0):
        assert mu > 0
        assert std > 0

        self.mu = mu
        self.std = std
        self.lower = self.mu * 0.5
        self.upper = self.mu * 1.8

        self.volume_condition: float = self._generate_volume()
        self.reset()

      def _generate_volume(self):
        v = random.gauss(self.mu, self.std)
        clipped = self.lower if v < self.lower else self.upper if v > self.upper else v
        return clipped


      def __call__(self, is_trade: bool, event: Union[Trade, OrderBook]):
        if is_trade and event.symbol == 'XBTUSD':
          self.volume += event.volume
          if self.volume > self.volume_condition:
            self.volume %= self.volume_condition
            self.volume_condition = self._generate_volume()
            return True
        return False

      def reset(self):
        self.volume = 0.0

    ################    RL agent operations wrapper    ################
    class Agent:
      def __init__(self, model: DuelingDQN, target: DuelingDQN,
                   condition: DecisionCondition,
                   gamma=0.98, lr=1e-3,
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
        self.EPS_END = 0.05
        self.EPS_DECAY = 250

        self.gamma = gamma
        self.loss = torch.nn.SmoothL1Loss()
        self.losses = []
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, amsgrad=True)

        self._update_each = update_each
        self.episode_counter = 0

        self.no_action_event = deque(maxlen=20)
        self.reset_state()
        self._is_training: bool = True

      def train(self):
        self._is_training = True
        self._model.train()
        self._target.eval()

      def eval(self):
        self._is_training = False
        self._model.eval()
        self._target.eval()

      def store_no_action(self, ts, price):
        self.no_action_event[0].append((ts, price))

      def episode_results(self) -> pd.DataFrame:
        fnames, states = agent.episode_files, agent.end_episode_states
        states = [t[0].tolist() + [t[1], t[2]] for t in states]
        episodes = [list(t[0]) + t[1] for t in zip(fnames, states)]
        res = pd.DataFrame(episodes, columns=['ob_file', 'tr_file', 'usd', 'xbt', 'eth', 'xbt_price', 'pennies'])
        res['nav'] = res.usd + res.xbt_price * res.xbt
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

        state_delta = torch.abs(prev_ps) - torch.abs(ps)
        pos = (torch.exp(b * tau) * state_delta)

        accumulation_penalty = -torch.exp(torch.abs(ps) * scale) + offset

        # a(V_t - V_{t-1}) + e^{b*tau} * sgn(|i_t| - |i_{t-1}|)
        return vv + pos + accumulation_penalty

      def get_new_reward(self, ps, ps_new, price, sigma, scale_rav=1.0/4, a=1.0):
        g1 = self.get_risk_adjusted_state_value(ps, price, sigma, scale=scale_rav)
        g2 = self.get_risk_adjusted_state_value(ps_new, price, sigma, scale=scale_rav)
        return a * (g2 - g1)

      def get_risk_adjusted_state_value(self, ps: torch.FloatTensor, price: torch.FloatTensor, sigma: torch.FloatTensor,
                                        tau=1.0, lambd=0.2, scale=1.0 / 10):
        nav = (ps[:, 0] + ps[:, 1]) * price  # updated to dollar NAV
        risk = torch.abs(ps[:, 1] * price) * tau * sigma  # updated here: sqrt of tau ; N(1, 1) * tau, sigma \in 1e-5

        return (nav - lambd * risk) * scale

      def get_terminal_reward(self, terminal_ps, terminal_prices, alpha=3.0, scale=4.0, shift=8.0):
        inbalance = (terminal_ps[:, 0] / terminal_prices - terminal_ps[:, 1])
        return torch.exp(torch.abs(inbalance) * scale) * torch.sign(inbalance) / shift  # 0: usd, 1: xbtusd, 2: ethusd

      def store_episode(self, new_obs, done, action, prices):
        if self._is_training:
          nans = [np.isnan(t).any() for t in [new_obs, done, action, prices]]
          if not any(nans):
            if self.is_initialized():
              self._replay_buffer.append((self.obs, self.action, new_obs, done, prices))
            self.obs = new_obs
            self.action = action
            self.prices = prices

      def is_initialized(self):
        return self.obs is not None  # happens after reset, first obs is missing

      def get_action(self, obs):
        if self._is_training:
          eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * self.episode_counter / self.EPS_DECAY)
          if np.random.uniform(0.0, 1.0) < eps :
            return random.randint(0, self._model.output_dim - 1)
        with torch.no_grad():
          qvals = self._target(torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0))[2]
          return torch.argmax(qvals).cpu().detach().item()

      def update(self):
        if len(self._replay_buffer) > self._batch_size and self._is_training:
          items = random.sample(self._replay_buffer, self._batch_size)
        # if self._is_training and len(self._replay_buffer) > 800:
        #   items = list(self._replay_buffer)[-800:]
          obs, action, next_obs, done, price = zip(*items)
          obs, next_obs, price = map(lambda x: torch.tensor(x, dtype=torch.float).to(device), [obs, next_obs, price])
          done = torch.tensor(done, dtype=torch.bool).to(device)
          action = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(device)

          ps, rs            = obs[:, -3:-1], obs[:, -1]
          next_ps, next_rs  = next_obs[:, -3:-1], next_obs[:, -1]

          qvalues = self._model(obs)[2]
          next_qvalues = self._target(next_obs[~done])[2]
          next_qvalues = next_qvalues.max(1)[0]

          # rs = rs[:, 2, 0] # all observations, midprice, smallest volume - here is always XBTUSD
          rewards = torch.zeros(size=done.shape, dtype=torch.float, device=device)
          # rewards[done] = self.get_terminal_reward(next_ps[done], meta[done][:, 0])
          # rewards[~done] = self.get_reward(v[~done], next_v, ps[~done][:, 1], next_ps[~done][:, 1], meta[~done][:, 1])
          rewards[~done] = self.get_new_reward(ps[~done], next_ps[~done], price[~done], rs[~done])
          rewards[done]  = self.get_risk_adjusted_state_value(next_ps[done], price[done], rs[done])

          rewards = rewards.clamp(-1.5, 1.5)  # reward clipping

          qvalues = qvalues.gather(1, action).squeeze(1)

          expected_q = torch.zeros(size=done.shape, dtype=torch.float, device=device)
          expected_q[~done] = self.gamma * next_qvalues
          expected_q += rewards

          loss = self.loss(qvalues, expected_q)  # or hubert loss
          self.losses.append(loss.detach().cpu().sum().item())
          if torch.isinf(loss).any() or torch.isnan(self._model.feature_layer[0].weight).any() or np.isnan(self.losses[-1]).any():
            print('nan is here')
          self.optimizer.zero_grad()
          loss.backward()
          for param in self._model.parameters():
            if torch.isinf(param.grad.data).any() or torch.isnan(param.grad.data).any():
              print('nan in grads')
            param.grad.data.clamp_(-1., 1.)  # gradient clipping
          self.optimizer.step()

    ################    Strategy wrapper    ################
    class RLStrategy(Strategy):
      def __init__(self, agent: Agent, rs: 'VWAP_modification', episode_length: int, cancels_enabled=False, **kwags):
        super().__init__(**kwags)
        self.agent: Agent = agent
        self.rs: VWAP_modification = rs
        # self._simulation_end: datetime.datetime = simulation_end
        self.action_space: Dict[int, Tuple[int, int]] = {
          0: (-1, 0),
          1: (0, -1),
          2: (-1, -1),
          3: (0, 0),
          4: (0, 1),
          5: (1, 0),
          6: (1, 1),
          7: (-1, 1),
          8: (1, -1),
        }
        self.cancels_enabled = cancels_enabled
        self.no_action_event = []
        self._episode_length = episode_length
        self.prices = None

      def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
        super().return_unfinished(statuses, memory)
        rs = self.rs.reset('XBTUSD')[0]
        obs, ps, prices = self.get_observation(memory, rs)
        self.agent.store_episode(obs, True, 0, prices)
        self.agent.end_episode_states.append((ps, prices, self.pennies))  # end state and prices
        self.agent.reset_state()

      def get_price_change(self, price) -> np.array:
        storage1: List[Trade] = self.metrics_map['trade-metric-120'].storage[('XBTUSD', 0)]
        storage2: List[Trade] = self.metrics_map['trade-metric-60'].storage[('XBTUSD', 0)]

        prices1 = np.array([t.price for t in storage1])
        prices2 = np.array([t.price for t in storage2])

        pc1 = (price - prices1.mean()) / 0.5  # xbtusd price step = 0.5 USD
        pc2 = (price - prices2.mean()) / 0.5  # xbtusd price step = 0.5 USD
        pc = np.array([pc1, pc2], dtype=np.float)

        return np.sign(pc) * np.log(np.abs(pc) + 1)

      def get_observation(self, memory, rs):
        items = list(map(lambda name: self.metrics_map[name].to_numpy('XBTUSD'), names + time_names))

        prices = self.get_prices(memory)
        pc = self.get_price_change(prices) # price change
        items = np.concatenate([t.flatten() for t in items + [pc]]).clip(-5.0, 5.0)

        ps = self.get_state()

        state = np.array([ps[0] / prices, ps[1], rs], dtype=np.float)
        items = np.concatenate([items, state], axis=None)
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
        t = ((self._simulation_end - ts).total_seconds() % self._episode_length) / self._episode_length
        return (t - 0.5) * 2
        # 10 minute episodes

      def action_to_order(self, action: int, memory, ts, quantity: int) -> List[OrderRequest]:
        offset_bid, offset_ask = self.action_space[action]

        offset_bid *= 0.5  # price step is .5 dollars
        offset_ask *= 0.5

        ob: OrderBook = memory[('orderbook', 'XBTUSD')]
        orders = []

        if offset_bid >= 0:
          orders.append(OrderRequest.create_bid(ob.bid_prices[0] - offset_bid, quantity, 'XBTUSD', ts))
        if offset_ask >= 0:
          orders.append(OrderRequest.create_ask(ob.ask_prices[0] + offset_ask, quantity, 'XBTUSD', ts))

        return orders

      def cancel_orders(self, statuses: List[OrderStatus], ts: datetime.datetime) -> List[OrderRequest]:
        statuses_ids = list(map(lambda x: x.id, statuses))
        active_ids = self.active_orders.keys()
        active_non_present = filter(lambda x: x not in statuses_ids, active_ids)
        return list(map(lambda x: OrderRequest.cancelOrder(self.active_orders[x], ts), active_non_present))

      def define_orders(self, row: Union[Trade, OrderBook],
                        statuses: List[OrderStatus],
                        memory: Dict[str, Union[Trade, OrderBook]],
                        is_trade: bool) -> List[OrderRequest]:
        if self.agent.condition(is_trade, row):
          orders = []
          cancels = []
          if self.cancels_enabled:
            cancels = self.cancel_orders(statuses, row.timestamp)
            orders += cancels

          rs = self.rs.reset('XBTUSD')[0]
          with CancelsApplied(self, cancels):
            obs, ps, prices = self.get_observation(memory, rs)
            action = self.agent.get_action(obs)
            if action == 2:
              self.agent.store_no_action(row.timestamp, prices)
            self.agent.store_episode(obs, False, action, prices)

          self.agent.update()

          orders += self.action_to_order(action, memory, row.timestamp, 1000)

        else:
          orders = []
        return orders

    class VWAP_modification(VWAP_volume):
      def __init__(self, T=10, **kwargs):
        super().__init__(**kwargs)
        self.storage = defaultdict(lambda: deque(maxlen=T))
        self.low   = defaultdict(lambda: None)
        self.high =  defaultdict(lambda: None)
        self.close = defaultdict(lambda: None)
        self.open =  defaultdict(lambda: None)
        self.last =  defaultdict(lambda: None)

      '''
      VWAP modification: returns vwap_offset and vwap_spread instead of bid,ask,midprice
      
      Also computes RS volatility
      '''
      def _evaluate(self, snapshot: OrderBook) -> np.array:
        vwap: np.array = super()._evaluate(snapshot)
        vwap_spread = (vwap[1,:] - vwap[0,:]) - 0.5  # vwap_ask - vwap_bid - default_spread
        midprice = (snapshot.bid_prices[0] + snapshot.ask_prices[0]) / 2
        vwap_offset = np.abs(vwap[[0,1], :] - midprice) - 0.25

        self.process(vwap[2, :], snapshot.symbol) # compute RS based on vwap_midprices

        return np.log(np.vstack([vwap_offset, vwap_spread]) + 1) # 3 rows

      def process(self, result: np.array, symbol: str):
        if self.open[symbol] is None:
          self.open[symbol] = result
          self.low[symbol] = result
          self.high[symbol] = result
        else:
          self.low[symbol] = np.minimum(self.low[symbol], result)
          self.high[symbol] = np.maximum(self.high[symbol], result)

        self.last[symbol] = result

      def reset(self, symbol: str) -> np.array:
        self.close[symbol] = self.last[symbol]

        sigma_sq = np.log(self.high[symbol] / self.close[symbol]) * np.log(self.high[symbol] / self.open[symbol])\
                   + np.log(self.low[symbol] / self.close[symbol]) * np.log(self.low[symbol] / self.open[symbol])
        self.storage[symbol].append(sigma_sq)

        self.low[symbol] = self.last[symbol]
        self.high[symbol] = self.last[symbol]
        self.open[symbol] = self.last[symbol]

        return np.sqrt(np.sum(self.storage[symbol], axis=0) / len(self.storage[symbol]) + 1e-8) + 1e-4


    ################    Comparison btw Strategies ############
    def compare_strategies(s1: RLStrategy, s2: GatlingMM, orderbook_file: str, trade_file: str,
                           delay=1) -> Tuple:

      r1 = TimeLimitedReader(orderbook_file, limit_time='100 min', trades_file=trade_file, nrows=500000)
      r2 = TimeLimitedReader(orderbook_file, limit_time='100 min', trades_file=trade_file, nrows=500000)

      o1 = SimulatedOrdersOutput()
      o2 = SimulatedOrdersOutput()

      s1.balance_listener = o1.balances.append
      s2.balance_listener = o2.balances.append

      b1 = Backtest(r1, s1, o1, delay=delay, warmup=True, stale_depth=8)
      b2 = Backtest(r2, s2, o2, delay=delay, warmup=False, stale_depth=2)

      b1.run()
      b2.run()

      return o1, o2

    def prepare_rl_strategy(agent: Agent, cancels_enabled=True) -> RLStrategy:
      vwap = VWAP_modification(T=20, volumes=[5e5, 1e6], name='vwap', z_normalize=1000)
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
      ], seconds=60, z_normalize=2500)
      trade_metric2 = TradeMetric(defaults, [
        # ('quantity', lambda x: len(x)),
        ('total', lambda trades: sum(map(lambda x: x.volume, trades)))
      ], seconds=120, z_normalize=2500)

      lipton_levels = 6
      hy = HayashiYoshido(140, True)
      lipton = Lipton(hy.name, lipton_levels)
      return RLStrategy(agent, vwap, episode_length=600,
                            instant_metrics=[vwap, liq],
                            delta_metrics=[hy],
                            time_metrics_trade=[trade_metric, trade_metric2], composite_metrics=[lipton],
                            initial_balance=0.0, cancels_enabled=cancels_enabled)


    ################    Simulation wrapper    ################
    def init_simulation(agent: Agent, orderbook_file: str, trade_file: str,
                        output_required: Union[bool, Output] = False, delay=5,
                        cancels_enabled=True) -> Optional[Output]:

      if isinstance(output_required, bool) and output_required:
        output = SimulatedOrdersOutput()
      elif isinstance(output_required, Output):
        output = output_required
      else:
        output = None

      reader = OrderbookReader(orderbook_file, trade_file, nrows=None, is_precomputed=True)

      strategy = prepare_rl_strategy(agent, cancels_enabled)
      backtest = BacktestOnSample(reader, strategy, output=output, delay=delay, warmup=True, stale_depth=8)
      backtest.run()

      return backtest.output

    ################    Constants    ################
    # matrices: [2, 4, 2], [2,3,2], [2, 2], [2]
    names = ['vwap', 'liquidity-spectrum', 'hayashi-yoshido', 'lipton']
    # [2, 2], [2, 2]
    time_names = ['trade-metric-60', 'trade-metric-120']

    ### Initialize agent, model, target network, decision unit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    condition = DecisionCondition(250000.0)
    model: DuelingDQN = DuelingDQN(input_dim=24, output_dim=7)
    target: DuelingDQN = DuelingDQN(input_dim=24, output_dim=7)
    # model.load_state_dict(torch.load('model-latest.pth'))
    model.train()
    target.eval()
    target.load_state_dict(model.state_dict())

    # target.load_state_dict(torch.load('notebooks/model.pth'))
    model.to(device)
    target.to(device)
    agent = Agent(model, target, condition, batch_size=8)

    ################  Initialize simulation  ################
    pairs = list(zip(glob.glob('../notebooks/time-sampled-10min/orderbook_*'), glob.glob('../notebooks/time-sampled-10min/trade_*')))
    # ob_file, tr_file = '../notebooks/time-sampled-10min/orderbook_1124.csv.gz', '../notebooks/time-sampled-10min/trade_1124.csv.gz'
    for i in range(2):
      ob_file, tr_file = random.choice(pairs)

    #   ob_file, tr_file = '../notebooks/time-sampled-10min/orderbook_1376.csv.gz', '../notebooks/time-sampled-10min/trade_1376.csv.gz'

      # trade_4020
      result_output = init_simulation(agent, ob_file, tr_file, output_required=True)
      agent.episode_files.append((ob_file, tr_file))
      if i == 8:
        print('8')

    ################  Visualize results  ################
    orders_side1, orders_side2 = list(result_output.orders.values())
    no_actions = list(agent.no_action_event[-2])
    fig, axs = make_plot_orderbook_trade(ob_file, 'XBTUSD',
                              orders_side1 + orders_side2,
                              no_actions,
                              orderbook_precomputed=True)
    plt.show()

    res = agent.episode_results()
    print(res[['nav', 'usd', 'xbt', 'xbt_price', 'pennies']])
