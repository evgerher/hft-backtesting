from abc import abstractmethod, ABC
from collections.__init__ import defaultdict, deque
from typing import List, Tuple, Union

import numpy as np

from hft.units.metric import Metric
from hft.utils.data import OrderBook, Trade
from hft.utils.types import Delta, DepleshionReplenishmentSide


class InstantMetric(Metric):
  def evaluate(self, arg: Union[Trade, OrderBook]) -> Union[np.array, float]:
    latest = self._evaluate(arg)
    self.latest[arg.symbol] = latest
    return latest

  @abstractmethod
  def _evaluate(self, *args) -> Union[np.array, float]:
    raise NotImplementedError

  def label(self) -> List[str]:
    return [self.name]

class InstantMultiMetric(InstantMetric):
  def __init__(self, name):
    super().__init__(name)
    subitems = self.subitems()
    # self.latest = {name: defaultdict(lambda: None) for name in subitems}
    self.latest = defaultdict(lambda: None)


  def evaluate(self, snapshot: OrderBook) -> List[np.array]:
      latest = self._evaluate(snapshot)
      for idx, item in enumerate(self.subitems()):
        self.latest[snapshot.symbol, item] = latest[idx]
      return latest

  @abstractmethod
  def subitems(self) -> List[str]:
    raise NotImplementedError

  def label(self):
    return [self.name] + self.subitems()


class _VWAP(InstantMultiMetric):
  def subitems(self) -> List[str]:
    return ['bid', 'ask', 'midpoint']

  def _evaluate(self, snapshot: OrderBook) -> Tuple[np.array, np.array, np.array]:
    vwap_bid, vwap_ask = self._bid(snapshot), self._ask(snapshot)
    midpoint = self._midpoint(vwap_bid, vwap_ask)
    return (vwap_bid, vwap_ask, midpoint)

  @abstractmethod
  def _evaluate_side(self, prices: np.array, volumes: np.array) -> np.array:
    raise NotImplementedError

  def _bid(self, snapshot: OrderBook) -> np.array:
    return self._evaluate_side(snapshot.bid_prices, snapshot.bid_volumes)

  def _ask(self, snapshot: OrderBook) -> np.array:
    return self._evaluate_side(snapshot.ask_prices, snapshot.ask_volumes)

  def _midpoint(self, vwap_bid: np.array, vwap_ask: np.array) -> np.array:
    return (vwap_bid + vwap_ask) / 2


class VWAP_depth(_VWAP):
  def __str__(self):
    return f'VWAP (Depth): {self.level}'

  def __init__(self, name = 'vwap-depth', level = 3):
    self.level = level
    super().__init__(name)

  def subitems(self):
    return [self.level]

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> np.array:
    # volumes are assumed to be sorted
    counter = 0
    i = 0
    while i < len(volumes):
      if volumes[i] != 0:
        counter += 1

      if counter == self.level:
        break

      i += 1

    volume_sum = np.sum(volumes[:i])
    return (prices[:i] * (volumes[:i] / volume_sum)).sum()


class VWAP_volume(_VWAP):

  def __str__(self):
    return f'<VWAP (Volume): {self.volumes}>'

  def __init__(self, volumes: List[int], symbol: str = None, name: str = 'vwap-volume_total'):
    self.volumes = sorted(volumes)
    self.symbol = symbol
    super().__init__(name)

  # def subitems(self):
  #   # return self.volumes
  #   return ['bid', 'ask', 'midprice']

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> np.array:
    i = 0
    values = np.zeros((len(self.volumes),), dtype=np.float)
    weights = np.zeros((len(volumes), ), dtype=np.float)

    sum_ = 0.0
    for idx, volume in enumerate(self.volumes):
      while sum_ < volume and i + 1 <= len(volumes):
        left = volume - np.sum(weights[:i])
        right = volumes[i]
        volume_taken = min(left, right)
        weights[i] = volume_taken
        i += 1
        sum_ = np.sum(weights)

      values[idx] = np.sum(prices[:i] * (weights[:i] / volume))
      i -= 1
    return values




class LiquiditySpectrum(InstantMetric):
  '''
  Implements Liquidity Spectrum features.
  Provides stregthed values of level volumes by summing up several of them. Another smoothing metric

  ls1 = sum(volumes[:3])
  ls2 = sum(volumes[3:6])
  ls3 = sum(volumes[6:])
  result = [ls1, ls2, ls3]
  '''
  def __init__(self):
    super().__init__(name='liquidity-spectrum')

  def _evaluate(self, orderbook: OrderBook) -> np.array:
    volumes = np.stack([orderbook.ask_volumes, orderbook.bid_volumes])
    ls1 = np.sum(volumes[:, :3], axis=1)
    ls2 = np.sum(volumes[:, 3:6], axis=1)
    ls3 = np.sum(volumes[:, 6:], axis=1)
    return np.stack([ls1, ls2, ls3])


class DeltaMetric(InstantMetric, ABC):
  def filter(self, event: Delta):
    return np.sum(event[-1][1, :]) != 0.0

  def evaluate(self, delta: Delta)  -> float:
    latest = self._evaluate(delta)
    self.latest[delta[1]] = latest
    return latest


class HayashiYoshido(DeltaMetric):
  # todo: check twice implementation
  '''
  Implements Hayashi-Yoshido UHF volatility estimator on log(!) values of deltas
  Accumulates updates on each step and automatically adjusts values of equation without need of reevaluation of whole sequence
  Denominator is `current_sum`, delimeters are `current_sq`

  Old values are removed via `sum_deque` and `sq_deque`, which keep track of affect on removal of old values.

  This Time metric does not provide `storage` field
  '''
  def __init__(self, seconds=60):
    super().__init__(name='hayashi-yoshido')
    self.seconds = seconds
    self.current_sum = {s: defaultdict(lambda: 0.0) for s in DepleshionReplenishmentSide}
    self.current_sq = {s: defaultdict(lambda: {True: 0.0, False: 0.0}) for s in DepleshionReplenishmentSide}

    self.current_p = {s: defaultdict(lambda: {True: None, False: None}) for s in DepleshionReplenishmentSide}
    self.last_diff = {s: defaultdict(lambda: {True: None, False: None}) for s in DepleshionReplenishmentSide}

    self.sum_deque = {s: defaultdict(lambda: deque()) for s in DepleshionReplenishmentSide}
    self.sq_deque = {s: defaultdict(lambda: {True: deque(), False: deque()}) for s in DepleshionReplenishmentSide}

    ## Main and Aux axes
    self.p1 = True
    self.p2 = False

  def __str__(self):
    return f'hayashi-yoshido-vol:{self.seconds}'

  def evaluate(self, delta: Delta) -> float:
    latest, queue = self._evaluate(delta)
    self.latest[delta[1], queue.name] = latest
    return latest, queue.value

  def _evaluate(self, event:Delta) -> Tuple[float, DepleshionReplenishmentSide]:

    def remove_old_values(symbol, timestamp):
      while len(sum_deque[symbol]) > 0 and (timestamp - sum_deque[symbol][0][0]).seconds > self.seconds:
        _, value = sum_deque[symbol].popleft()
        current_sum[symbol] -= value

      for p in [self.p1, self.p2]:
        coll = sq_deque[symbol][p]
        while len(coll) > 0 and (timestamp - coll[0][0]).seconds > self.seconds:
          _, value = coll.popleft()
          current_sq[symbol][p] -= value

    def evaluate_side(value):
      if current_p[symbol][p] is None:
        current_p[symbol][p] = value
      else:
        diff = value - current_p[symbol][p]
        last_diff[symbol][p] = diff
        current_p[symbol][p] = value

        sq_update = diff ** 2
        current_sq[symbol][p] += sq_update
        sq_deque[symbol][p].append((ts, sq_update))

        if last_diff[symbol][not p] is not None and current_sq[symbol][not p] > 0:
          sum_update = diff * last_diff[symbol][not p]
          current_sum[symbol] += sum_update
          sum_deque[symbol].append((ts, sum_update))

          return current_sum[symbol] / np.sqrt(current_sq[symbol][p]) / np.sqrt(current_sq[symbol][not p])
      return None

    sign, quote_side = np.sum(event[-1][1, :]), event[2] # get first volume from `price-volume` np array
    ts, symbol = event[0], event[1]
    value = np.sum(event[-1][1, :])
    p = value > 0
    value = np.log(abs(value))

    queues: DepleshionReplenishmentSide = DepleshionReplenishmentSide.eval(sign, quote_side)
    current_p = self.current_p[queues]
    current_sum = self.current_sum[queues]
    sum_deque = self.sum_deque[queues]
    current_sq = self.current_sq[queues]
    last_diff = self.last_diff[queues]
    sq_deque = self.sq_deque[queues]

    remove_old_values(symbol, ts)
    return evaluate_side(value), queues
