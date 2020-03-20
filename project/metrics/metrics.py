from dataclasses import dataclass
from typing import List, Deque, Dict, Callable, Tuple, Union, Sequence, TypeVar
import datetime
from collections import deque, defaultdict

from metrics.types import Delta, DeltaValue, NamedExecutable, DeltaExecutable, TradeExecutable
from utils.data import OrderBook, Trade
import numpy as np
import math
from abc import ABC, abstractmethod

@dataclass
class MetricData:
  name: str
  symbol: str
  value: float


class Metric(ABC):

  def __init__(self, name):
    self.name = name  # map of name -> metric for dependency injection
    self.latest = defaultdict(lambda: None)

  # def evaluate(self, *args) -> 'MetricData':
  @abstractmethod
  def evaluate(self, *args):
    raise NotImplementedError

  @abstractmethod
  def label(self):
    raise NotImplementedError

class InstantMetric(Metric):
  def evaluate(self, *args)  -> Union[np.array, float]:
    latest = self._evaluate(args)
    self.latest = latest
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
    self.latest = {name: defaultdict(lambda: None) for name in subitems}

  def evaluate(self, snapshot: OrderBook) -> List[np.array]:
      latest = self._evaluate(snapshot)
      for idx, item in enumerate(self.subitems()):
        self.latest[item] = latest[idx]
      return latest

  @abstractmethod
  def subitems(self) -> List[str]:
    raise NotImplementedError

  def label(self):
    return [self.name] + self.subitems()

class _VWAP(InstantMultiMetric):
  def names(self) -> List[str]:
    return [f'{self.__str__()} bid', f'{self.__str__()} ask', f'{self.__str__()} midpoint']

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
    return f'VWAP (Depth): {self.levels}'

  def __init__(self, name = 'vwap-depth', levels = 3):
    super().__init__(name)
    self.levels = levels

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> np.array: # todo: test
    # volumes are assumed to be sorted
    counter = 0
    i = 0
    while i < len(volumes):
      if volumes[i] != 0:
        counter += 1

      if counter == self.levels:
        break

      i += 1

    volume_sum = np.sum(volumes[:i])
    return (prices[:i] * (volumes[:i] / volume_sum)).sum()

class VWAP_volume(_VWAP):

  def __str__(self):
    return f'<VWAP (Volume): {self.volumes}>'

  def __init__(self, volumes: List[int], symbol: str = None, name: str = 'vwap-volume'):
    super().__init__(name)
    self.volumes = sorted(volumes)
    self.symbol = symbol

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

class TimeMetric(Metric):
  def __init__(self, name,
               callables: List[NamedExecutable],
               seconds=60,
               starting_moment: datetime.datetime = None):

    super().__init__(name)
    self.metric_names: List[str] = [c[0] for c in callables]
    self.seconds = seconds
    # symbol, side -> trade
    self._storage: Dict[Tuple, Deque] = defaultdict(deque)
    self._callables: List[Callable[[List], float]] = [c[1] for c in callables]
    self._from: datetime.datetime = starting_moment
    self._skip_from = False

  def _skip(self, timestamp):
    if (timestamp - self._from).seconds >= self.seconds:
      self._skip_from = True
    return [-1.0] * len(self._callables)


  def label(self):
    return [self.name] + self.metric_names

  @abstractmethod
  def _get_update_deque(self, event):
    raise NotImplementedError

  def evaluate(self, event: Trade) -> List[float]:
    target: Deque = self._get_update_deque(event)

    if not self._skip_from:
      return self._skip(event.timestamp)
    else:
      self._remove_old_values(event, target)

      metrics = []
      for _callable in self._callables:
        metrics.append(_callable(target))  # order matters

      names = self.metric_names
      assert len(names) == len(metrics)
      for idx, name in enumerate(names):
        self.latest[name] = metrics[idx]
      return metrics

  @abstractmethod
  def _remove_old_values(self, event, storage: Deque[Union[Trade, Delta]]):
    raise NotImplementedError

  def set_starting_moment(self, moment: datetime.datetime):
    self._from = moment


class TradeMetric(TimeMetric):
  def __init__(self, name: str,
               callables: List[TradeExecutable],
               seconds=60):
    super().__init__(f'{name}-{seconds}', callables, seconds)

  def _remove_old_values(self, event: Trade, storage: Deque[Trade]):
    while (event.timestamp - storage[0].timestamp).seconds > self.seconds:
      storage.popleft()

  def _get_update_deque(self, event: Trade):
    key = (event.symbol, event.side)
    target: Deque[Trade] = self._storage[key]
    target.append(event)
    return target

  def __str__(self):
    return f'trade-time-metric:{self.seconds}'


class DeltaMetric(TimeMetric):
  def __init__(self,
               seconds=60,
               starting_moment: datetime.datetime = None):

    callables: List[DeltaExecutable] = [('Delta',) ] # todo this
    super().__init__(f'delta-{seconds}', callables, seconds, starting_moment)
    self._time_storage = defaultdict(deque)

  def _remove_old_values(self, event: Delta, storage: Deque[Tuple[datetime.datetime, int]]):
    timestamp = event[0]
    symbol = event[1]
    side = event[2]
    volume = event[3]
    sign = 'pos' if volume > 0 else 'neg'
    key = (symbol, side, sign)

    time_storage = self._time_storage[key]
    while (event[0] - time_storage[0]).seconds > self.seconds:
      time_storage.popleft()
      storage.popleft()

  def _get_update_deque(self, event: Delta):
    timestamp = event[0]
    symbol = event[1]
    side = event[2]
    volume = event[3]
    sign = 'pos' if volume > 0 else 'neg'

    key = (symbol, side, sign)
    target: Deque[int] = self._storage[key]
    self._time_storage[key].append(timestamp)
    target.append(volume)
    return target

  def __str__(self):
    return f'delta-time-metric:{self.seconds}'

class CompositeMetric(InstantMetric):
  def __init__(self, name: str, metric_map: Dict[str, Metric] = None):
    super().__init__(name)
    self._metric_map = metric_map

  def set_metric_map(self, metric_map: Dict[str, Metric]):
    self._metric_map = metric_map

class Lipton(CompositeMetric):
  def __init__(self, metric_map: Dict[str, Metric]):
    super().__init__('lipton-upward-probability', metric_map)

  def _evaluate(self, snapshot: OrderBook): # todo: test it
    delta = self._metric_map['delta-60'].latest
    replenishment_ask: List[int] = delta[(snapshot.symbol, 'ask', 'pos')]
    depletion_bid: List[int] = delta[(snapshot.symbol, 'bid', 'neg')]
    p_xy = np.corrcoef(depletion_bid, replenishment_ask)[0, 1]

    x = float(snapshot.bid_volumes[0])
    y = float(snapshot.ask_volumes[0])
    sqrt_corr = np.sqrt((1 + p_xy) / (1 - p_xy))
    p = 0.5 * (1. - np.arctan(sqrt_corr * (y - x) / (y + x)) / np.arctan(sqrt_corr))
    return p

  def bidask_imbalance(self, snapshot: OrderBook):
    q_b = snapshot.bid_volumes[0]
    q_a = snapshot.ask_volumes[0]
    imbalance = (q_b - q_a) / (q_b + q_a)
    return MetricData('bidask-imbalance', snapshot.symbol, imbalance)

  def label(self):
    return [self.name]


class QuickestDetection(CompositeMetric): # todo: does not work properly
  # todo: what if h1 and h2 are functions?
  def __init__(self, h1, h2, name: str, metric_map: Dict[str, Metric], time_horizon=600):
    super().__init__(name, metric_map)
    self.h1 = h1
    self.h2 = h2
    self._max = 0.0
    self._min = float('inf')
    self._values: List[float] = []
    self._dates: List[datetime.datetime] = []
    self.time_horizon = time_horizon
    self.__divisor = 4 * math.log(2)

  def _reset_state(self, price):
    self._max = price
    self._min = price

  # https://portfolioslab.com/rogers-satchell
  def roger_satchell_volatility(self):
    o, c = self._values[0], self._values[-1]
    h, l = max(self._values), min(self._values)

    # todo: does not take into consideration size of list
    return math.sqrt(math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l * o))

  def volatility_2(self):
    h, l = max(self._values), min(self._values)
    return math.sqrt(math.log(h / l) / self.__divisor)

  def _update(self, item: Tuple[float, datetime.datetime]):
    price = item[0]
    timestamp = item[1]
    self._dates.append(timestamp)
    self._values.append(price)

    if price > self._max:
      self._max = price

    if price < self._min:
      self._min = price

    while (timestamp - self._dates[0]).seconds > self.time_horizon:
      self._dates.pop(0)
      self._values.pop(0)

  def evaluate(self, item: Tuple[float, datetime.datetime]): # todo: accepts mid-point
    price     = item[0]
    self._update(item)

    sigma = self.roger_satchell_volatility()

    # check maximum condition
    upper_trend = self._max - price < self.h1 * sigma
    down_trend = price - self._min < self.h2 * sigma

    if upper_trend or down_trend:
      self._reset_state(price)

    if upper_trend and down_trend:
      raise ArithmeticError("IT IS NOT POSSIBLE")

    if upper_trend:
      return 1

    if down_trend:
      return -1

    return 0
