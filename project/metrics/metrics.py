from dataclasses import dataclass
from typing import List, Deque, Dict, Callable, Tuple
import datetime
from collections import deque, defaultdict
from utils.data import OrderBook, Trade
import numpy as np

@dataclass
class MetricData:
  name: str
  symbol: str
  value: float


class Metric:
  # def evaluate(self, *args) -> 'MetricData':
  def evaluate(self, *args) -> float:
      pass

class InstantMetric(Metric):
  def evaluate(self, *args) -> List[float]:
  # def evaluate(self, *args) -> List['MetricData']:

      pass

class ContinuousMetric:

  def __init__(self, n):
    self.n = n

  def evaluate(self, items):
    pass

class CompositeMetric:
  def __init__(self, *metric_names):
    self.targets = metric_names

  def evaluate(self):
    pass

class TimeMetric(Metric):
  def __init__(self, callables: List[Tuple[str, Callable[[List[Trade]], float]]],
               seconds=60,
               starting_moment: datetime.datetime = None):
    self.metric_names: List[str] = [f'{c[0]}_{seconds}' for c in callables]
    self.seconds = seconds
    self._storage: Dict[str, Deque[Trade]] = defaultdict(deque)
    self._callables: List[Callable[[List[Trade]], float]] = [c[1] for c in callables]
    self._from: datetime.datetime = starting_moment
    self._skip_from = False

  def evaluate(self, trade: Trade) -> List[float]:
    target: Deque[Trade] = self._storage[trade.label()]
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

  def evaluate(self, snapshot: OrderBook) -> List[float]:
    vwap_bid, vwap_ask = self.VWAP_bid(snapshot), self.VWAP_ask(snapshot)
    midpoint = self.VWAP_midpoint(vwap_bid, vwap_ask)
    # return (
    #   MetricData(f'{self.__str__()} bid', snapshot.symbol, vwap_bid),
    #   MetricData(f'{self.__str__()} ask', snapshot.symbol, vwap_ask),
    #   MetricData(f'{self.__str__()} midpoint', snapshot.symbol, midpoint)
    # )
    return [vwap_bid, vwap_ask, midpoint]

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float:
    pass

  def VWAP_bid(self, snapshot: OrderBook) -> float:
    return self._evaluate_side(snapshot.bid_prices, snapshot.bid_volumes)

  def VWAP_ask(self, snapshot: OrderBook) -> float:
    return self._evaluate_side(snapshot.ask_prices, snapshot.ask_volumes)

  def VWAP_midpoint(self, vwap_bid: float, vwap_ask: float) -> float:
    return (vwap_bid + vwap_ask) / 2

class VWAP_depth(_VWAP):
  def __str__(self):
    return f'VWAP (Depth): {self.levels}'

  def __init__(self, levels = 3):
    self.levels = levels

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float: # todo: test
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
    return f'<VWAP (Volume): {self.volume}>'

  def __init__(self, volume: int = 1e6, symbol: str = None):
    self.volume = volume
    self.symbol = symbol

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float: # todo: test
    total_volumes: int = 0
    i = 0

    weights = []
    while total_volumes < self.volume and i + 1 < len(volumes):
      _volume_taken = min(self.volume - total_volumes, volumes[i + 1])
      total_volumes += _volume_taken
      weights.append(_volume_taken * 1.0 / self.volume)
      i += 1

    weights = np.array(weights)
    return np.sum(prices[:len(weights)] * weights) / np.sum(weights)

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

class QuickestDetection(ContinuousMetric):
  def evaluate(self, items):
    pass
