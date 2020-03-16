from dataclasses import dataclass
from typing import List, Deque, Dict, Callable, Tuple, Union
import datetime
from collections import deque, defaultdict
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
  # def evaluate(self, *args) -> 'MetricData':
  @abstractmethod
  def evaluate(self, *args):
    raise NotImplementedError

class InstantMetric(Metric):
  @abstractmethod
  def evaluate(self, *args) -> Union[np.array, float]:
  # def evaluate(self, *args) -> List['MetricData']:
    raise NotImplementedError

  @abstractmethod
  def label(self) -> str:
    raise NotImplementedError

class ContinuousMetric(ABC):

  def __init__(self, n):
    self.n = n

  @abstractmethod
  def evaluate(self, items):
    not NotImplementedError

class CompositeMetric(ABC):
  def __init__(self, *metric_names):
    self.targets = metric_names

  @abstractmethod
  def evaluate(self):
    not NotImplementedError

class TimeMetric(Metric):
  def __init__(self, callables: List[Tuple[str, Callable[[List[Trade]], float]]],
               seconds=60,
               starting_moment: datetime.datetime = None):
    self.metric_names: List[str] = [f'{c[0]}_{seconds}' for c in callables]
    self.seconds = seconds
    # symbol, side -> trade
    self._storage: Dict[(str, str), Deque[Trade]] = defaultdict(deque)
    self._callables: List[Callable[[List[Trade]], float]] = [c[1] for c in callables]
    self._from: datetime.datetime = starting_moment
    self._skip_from = False

  def evaluate(self, trade: Trade) -> List[float]:
    target: Deque[Trade] = self._storage[(trade.symbol, trade.side)]
    target.append(trade)

    if not self._skip_from:
      if (trade.timestamp - self._from).seconds >= self.seconds:
        self._skip_from = True
      metrics = []
      # for _callable in self._callables:
      #   metrics.append(MetricData(f'TimeMetric {_callable[0]}', trade.label(), -1.0))
      return [-1.0] * len(self._callables)
    else:
      while (trade.timestamp - target[0].timestamp).seconds >= self.seconds:
        target.popleft()

      metrics = []
      for _callable in self._callables:
      #   metrics.append(MetricData(f'TimeMetric {_callable[0]}', trade.label(), _callable[1](target)))
        metrics.append(_callable(target))

      return metrics

  def set_starting_moment(self, moment: datetime.datetime):
    self._from = moment

  def __str__(self):
    return f'Time metric seconds={self.seconds}'

class _VWAP(InstantMetric):

  def names(self) -> List[str]:
    return [f'{self.__str__()} bid', f'{self.__str__()} ask', f'{self.__str__()} midpoint']

  def evaluate(self, snapshot: OrderBook) -> List[np.array]:
    vwap_bid, vwap_ask = self.VWAP_bid(snapshot), self.VWAP_ask(snapshot)
    midpoint = self.VWAP_midpoint(vwap_bid, vwap_ask)
    # return (
    #   MetricData(f'{self.__str__()} bid', snapshot.symbol, vwap_bid),
    #   MetricData(f'{self.__str__()} ask', snapshot.symbol, vwap_ask),
    #   MetricData(f'{self.__str__()} midpoint', snapshot.symbol, midpoint)
    # )
    return [vwap_bid, vwap_ask, midpoint]

  @abstractmethod
  def _evaluate_side(self, prices: np.array, volumes: np.array) -> np.array:
    raise NotImplementedError

  def VWAP_bid(self, snapshot: OrderBook) -> np.array:
    return self._evaluate_side(snapshot.bid_prices, snapshot.bid_volumes)

  def VWAP_ask(self, snapshot: OrderBook) -> np.array:
    return self._evaluate_side(snapshot.ask_prices, snapshot.ask_volumes)

  def VWAP_midpoint(self, vwap_bid: np.array, vwap_ask: np.array) -> float:
    return (vwap_bid + vwap_ask) / 2

class VWAP_depth(_VWAP):
  def __str__(self):
    return f'VWAP (Depth): {self.levels}'

  def __init__(self, levels = 3):
    self.levels = levels

  def label(self):
    return 'vwap-depth'

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

  def __init__(self, volumes: List[int], symbol: str = None):
    self.volumes = sorted(volumes)
    self.symbol = symbol

  def label(self):
    return f'vwap-volume {self.symbol}'

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> np.array: # todo: test
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

class Lipton(InstantMetric):
  def bidask_imbalance(self, snapshot: OrderBook):
    q_b = snapshot.bid_volumes[0]
    q_a = snapshot.ask_volumes[0]
    imbalance = (q_b - q_a) / (q_b + q_a)
    return MetricData('bidask-imbalance', snapshot.symbol, imbalance)

  def upward_probability(self, snapshot: OrderBook, p_xy=-0.5): # todo: should I consider depth or only best prices available
    """
    x - bid quote sizes
    y - ask quote sizes
    :param snapshot:
    :param p_xy: correlation between the depletion and replenishment of the bid and ask queues' diffusion processes (typically negative)
    :return:
    """
    # todo: how to evaluate p_xy ?
    # todo: implement p_xy
    x = snapshot.bid_volumes[0]
    y = snapshot.ask_volumes[0]
    sqrt_corr = np.sqrt((1 + p_xy) / (1 - p_xy))
    p = 1. / 2 * (1. - np.arctan(sqrt_corr * (y - x) / (y + x)) / np.arctan(sqrt_corr))
    return MetricData('Lipton-unward-probability', snapshot.symbol, p)

class QuickestDetection(InstantMetric):
  # todo: what if h1 and h2 are functions?
  def __init__(self, h1, h2, time_horizon=600):
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
