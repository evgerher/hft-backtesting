import datetime
from abc import abstractmethod, ABC
from collections import defaultdict, deque
from typing import List, Tuple, Union, Sequence

import numpy as np

from hft.units.metric import Metric
from hft.utils.data import OrderBook, Trade, Delta
from hft.utils.types import DepleshionReplenishmentSide


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

  def to_numpy(self) -> np.array:
    return np.array(list(self.latest.values()), dtype=np.float)


class _VWAP(InstantMetric):
  def subitems(self) -> List[str]:
    return ['bid', 'ask', 'midpoint']

  def _evaluate(self, snapshot: OrderBook) -> np.array:
    vwap_bid, vwap_ask = self._bid(snapshot), self._ask(snapshot)
    midpoint = self._midpoint(vwap_bid, vwap_ask)
    return np.stack([vwap_bid, vwap_ask, midpoint])

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

  def __init__(self, name = 'vwap-depth', level = 3, **kwargs):
    self.level = level
    defaults = [
      ('XBTUSD', np.zeros(shape=(3, level))),
      ('ETHUSD', np.zeros(shape=(3, level)))
    ]
    super().__init__(name, defaults, **kwargs)

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

  def __init__(self, volumes: List[int], symbol: str = None, name: str = 'vwap-volume', **kwargs):
    self.volumes = sorted(volumes)
    self.symbol = symbol
    defaults = [
      ('XBTUSD', np.zeros(shape=(3, len(volumes)))),
      ('ETHUSD', np.zeros(shape=(3, len(volumes))))
    ]
    super().__init__(name, defaults, **kwargs)

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

      weighted = weights / volume
      ratio = np.sum(weighted) # when there is not enough volume on known levels to meet condition
      values[idx] = np.sum(prices * weighted) / ratio
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
  def __init__(self, **kwargs):
    defaults = [
      ('XBTUSD', np.zeros((3, 2))),
      ('ETHUSD', np.zeros((3, 2))),
    ]
    super().__init__('liquidity-spectrum', defaults, **kwargs)

  def _evaluate(self, orderbook: OrderBook) -> np.array:
    volumes = np.stack([orderbook.ask_volumes, orderbook.bid_volumes])
    ls1 = np.sum(volumes[:, :3], axis=1)
    ls2 = np.sum(volumes[:, 3:6], axis=1)
    ls3 = np.sum(volumes[:, 6:], axis=1)
    return np.stack([ls1, ls2, ls3])


class DeltaMetric(InstantMetric, ABC):
  def filter(self, event: Delta):
    return np.sum(event.diff[1, :]) != 0.0

  def evaluate(self, delta: Delta)  -> float:
    latest = self._evaluate(delta)
    self.latest[delta.symbol] = latest
    return latest


class HayashiYoshido(DeltaMetric):
  '''
  Implements Hayashi-Yoshido UHF volatility estimator on log(!) values of deltas
  Accumulates updates on each step and automatically adjusts values of equation without need of reevaluation of whole sequence
  Denominator is `current_sum`, delimeters are `current_sq`

  Old values are removed via `sum_deque` and `sq_deque`, which keep track of affect on removal of old values.

  This Time metric does not provide `storage` field
  '''
  def __init__(self, seconds=140, normalization=False, **kwargs):

    defaults = []

    # todo: strictly requires refactoring!
    symbols = ['XBTUSD', 'ETHUSD']
    for symbol in symbols:
      for s in DepleshionReplenishmentSide:
        defaults.append(((symbol, s.name), 0.0))

    super().__init__('hayashi-yoshido', defaults, **kwargs)
    self.seconds = seconds

    self.__current_sum = {s: defaultdict(lambda: 0.0) for s in DepleshionReplenishmentSide}
    self.__current_sq = {s: defaultdict(lambda: {True: 0.0, False: 0.0}) for s in DepleshionReplenishmentSide}
    self.__last_diff = {s: defaultdict(lambda: {True: None, False: None}) for s in DepleshionReplenishmentSide}
    self._integral_sum = {s: defaultdict(lambda: {True: 0.0, False: 0.0}) for s in DepleshionReplenishmentSide}

    self._previous_t = {s: defaultdict(lambda: {True: None, False: None}) for s in DepleshionReplenishmentSide}

    self.__sum_deque = {s: defaultdict(deque) for s in DepleshionReplenishmentSide}
    self.__sq_deque = {s: defaultdict(lambda: {True: deque(), False: deque()}) for s in DepleshionReplenishmentSide}
    self._integral_deque = {s: defaultdict(lambda: {True: deque(), False: deque()}) for s in DepleshionReplenishmentSide}

    ## Main and Aux axes
    self.__p1 = True
    self.__p2 = False

    self._normalization = normalization

  def __str__(self):
    return f'hayashi-yoshido-vol:{self.seconds}'

  def evaluate(self, delta: Delta) -> Tuple[float, int]: # todo: explain
    latest, queue = self._evaluate(delta)
    self.latest[delta.symbol, queue.name] = latest
    return latest, queue.value

  def _evaluate(self, event:Delta) -> Tuple[float, DepleshionReplenishmentSide]:

    def remove_old_values(symbol, timestamp):
      while len(sum_deque[symbol]) > 0 and (timestamp - sum_deque[symbol][0][0]).seconds > self.seconds:
        _, value = sum_deque[symbol].popleft()
        current_sum[symbol] -= value

      for p in [self.__p1, self.__p2]:
        for coll, upd in zip((sq_deque[symbol][p], integral_deque[symbol][p]), (current_sq[symbol], integral_sum[symbol])):
          while len(coll) > 0 and (timestamp - coll[0][0]).seconds > self.seconds:
            _, value = coll.popleft()
            upd[p] -= value

    def evaluate_side(value):
      if last_diff[symbol][p] is None:
        last_diff[symbol][p] = value
        previous_t[symbol][p] = ts
      else:
        integral_square = (ts - previous_t[symbol][p]).total_seconds() * value / self.seconds
        integral_sum[symbol][p] += integral_square  # todo, make it optional action
        previous_t[symbol][p] = ts
        integral_deque[symbol][p].append((ts, integral_square))

        if self._normalization:
          diff = value - integral_sum[symbol][p] # mean
        else:
          diff = value
        last_diff[symbol][p] = diff

        sq_update = diff ** 2
        current_sq[symbol][p] += sq_update
        sq_deque[symbol][p].append((ts, sq_update))

        if last_diff[symbol][not p] is not None and current_sq[symbol][not p] > 0:
          sum_update = diff * last_diff[symbol][not p]
          current_sum[symbol] += sum_update
          sum_deque[symbol].append((ts, sum_update))

          return current_sum[symbol] / np.sqrt(current_sq[symbol][p]) / np.sqrt(current_sq[symbol][not p])
      return None

    value, quote_side = np.sum(event.diff[1, :]), event.quote_side # get first volume from `price-volume` np array
    ts, symbol = event.timestamp, event.symbol
    p = value > 0
    queues: DepleshionReplenishmentSide = DepleshionReplenishmentSide.eval(value, quote_side)
    value = abs(value)


    current_sum = self.__current_sum[queues]
    sum_deque = self.__sum_deque[queues]
    current_sq = self.__current_sq[queues]
    last_diff = self.__last_diff[queues]
    sq_deque = self.__sq_deque[queues]
    integral_sum = self._integral_sum[queues]
    integral_deque = self._integral_deque[queues]
    previous_t = self._previous_t[queues]

    remove_old_values(symbol, ts)
    return evaluate_side(value), queues


class CraftyCorrelation(DeltaMetric):
  def __init__(self, seconds:int, block_size:int, name, **kwargs):
    assert seconds % block_size == 0
    defaults = []

    # todo: strictly requires refactoring!
    symbols = ['XBTUSD', 'ETHUSD']
    for symbol in symbols:
      for s in DepleshionReplenishmentSide:
        defaults.append(((symbol, s.name), 0.0))

    super().__init__(name, defaults, **kwargs)
    self.seconds = seconds

    self._blocks_quantity = seconds // block_size
    self._block_time = datetime.timedelta(seconds=block_size)
    # sum and q in current block
    self._latest_block = {s: defaultdict(dict) for s in DepleshionReplenishmentSide}
    # avg over blocks
    self._blocks = {s: defaultdict(lambda: {True: deque(maxlen=self._blocks_quantity),
                                            False: deque(maxlen=self._blocks_quantity)})
                    for s in DepleshionReplenishmentSide}
    # timer for the next block, until it did not appear
    self._next_block_t = {s: defaultdict(lambda: {True: None, False: None}) for s in DepleshionReplenishmentSide}


  def evaluate(self, delta: Delta) -> Tuple[float, int]: # todo: explain
    latest, queue = self._evaluate(delta)
    if latest is None: # reload existing value
      latest = self.latest[delta.symbol, queue.name]
    else:
      self.latest[delta.symbol, queue.name] = latest
    return latest, queue.value

  def _evaluate(self, event: Delta):
    value, quote_side = np.sum(event.diff[1, :]), event.quote_side  # get first volume from `price-volume` np array
    ts, symbol = event.timestamp, event.symbol
    p = value > 0
    queues: DepleshionReplenishmentSide = DepleshionReplenishmentSide.eval(value, quote_side)
    value = abs(value)

    if self._next_block_t[queues][symbol][p] is None:
      self._next_block_t[queues][symbol][p] = ts + self._block_time
      self._latest_block[queues][symbol][p] = (value, 1)
      return None, queues
    else:
      if self._next_block_t[queues][symbol][p] < ts:
        # update queues
        cur_block = self._latest_block[queues][symbol][p]
        self._blocks[queues][symbol][p].append(1.0 * cur_block[0] / cur_block[1])
        self._latest_block[queues][symbol][p] = (value, 1)
        self._next_block_t[queues][symbol][p] = ts + self._block_time

        # eval correlation
        items = self._blocks[queues][symbol][p]
        vsitems = self._blocks[queues][symbol][not p]


        s1, s2 = map(np.array, [items, vsitems])
        l_items, l_vsitems = map(len, [items, vsitems])

        if all([l_vsitems > 1, l_vsitems > 1]):
          if l_items != l_vsitems:
            l = min(l_items, l_vsitems)
            s1 = s1[-l:]
            s2 = s2[-l:]

          s1 -= np.mean(s1)
          s2 -= np.mean(s2)

          cov = np.dot(s1, s2)
          corr = cov / np.linalg.norm(s1, 2) / np.linalg.norm(s2, 2)
          return corr, queues
        else:
          return None, queues
      else:
        cur_block = self._latest_block[queues][symbol][p]
        self._latest_block[queues][symbol][p] = (cur_block[0] + value, cur_block[1] + 1)
        return None, queues
