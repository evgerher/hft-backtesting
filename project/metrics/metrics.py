from dataclasses import dataclass
from typing import List, Deque, Dict, Callable, Tuple
import datetime
from collections import deque, defaultdict
from utils.data import Snapshot, Trade
import numpy as np

@dataclass
class MetricData:
  name: str
  symbol: str
  value: float


class Metric:
  def evaluate(self, *args) -> 'MetricData':
    pass

class InstantMetric(Metric):
  def evaluate(self, *args) -> List['MetricData']:
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
  def __init__(self, callable: List[Tuple[str, Callable[[List[Trade]], float]]], seconds=60, starting_moment: datetime.datetime = None):
    self._seconds = seconds
    self._storage: Dict[str, Deque[Trade]] = defaultdict(deque)
    self._callables: List[Tuple[str, Callable[[List[Trade]], float]]] = callable
    self._from: datetime.datetime = starting_moment
    self._skip_from = False

  def evaluate(self, trade: Trade) -> List[MetricData]:
    target: Deque[Trade] = self._storage[trade.label()]
    target.append(trade)

    if not self._skip_from:
      if (trade.timestamp - self._from).seconds > self._seconds:
        self._skip_from = True
      metrics = []
      for _callable in self._callables:
        metrics.append(MetricData(f'TimeMetric {_callable[0]}', trade.label(), None))
      return metrics
    else:
      while (trade.timestamp - target[0].timestamp).seconds > self._seconds:
        target.popleft()

      metrics = []
      for _callable in self._callables:
        metrics.append(MetricData(f'TimeMetric {_callable[0]}', trade.label(), _callable[1](target)))
      return metrics

  def set_starting_moment(self, moment: datetime.datetime):
    self._from = moment

  def __str__(self):
    return f'Time metric seconds={self._seconds}'

class _VWAP(InstantMetric):

  def evaluate(self, snapshot: Snapshot):
    vwap_bid, vwap_ask = self.VWAP_bid(snapshot), self.VWAP_ask(snapshot)
    midpoint = self.VWAP_midpoint(vwap_bid, vwap_ask)
    return (
      MetricData(f'{self.__str__()} bid', snapshot.symbol, vwap_bid),
      MetricData(f'{self.__str__()} ask', snapshot.symbol, vwap_ask),
      MetricData(f'{self.__str__()} midpoint', snapshot.symbol, midpoint)
    )

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float:
    pass

  def VWAP_bid(self, snapshot: Snapshot) -> float:
    return self._evaluate_side(snapshot.bid_prices, snapshot.bid_volumes)

  def VWAP_ask(self, snapshot: Snapshot) -> float:
    return self._evaluate_side(snapshot.ask_prices, snapshot.ask_volumes)

  def VWAP_midpoint(self, vwap_bid: float, vwap_ask: float) -> float:
    return (vwap_bid + vwap_ask) / 2

class VWAP_depth(_VWAP):
  def __str__(self):
    return f'VWAP (Depth): {self.levels}'

  def __init__(self, levels = 3):
    self.levels = levels

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float:
    # volumes are assumed to be sorted
    volume_sum = np.sum(volumes[:self.levels])
    return prices[:self.levels] * volumes[self.levels] / volume_sum

class VWAP_volume(_VWAP):

  def __str__(self):
    return f'<VWAP (Volume): {self.volume} for symbol: {self.symbol}>'

  def __init__(self, volume: int = 1e6, symbol: str = None):
    self.volume = volume
    self.symbol = symbol

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float:
    total_volumes: int = 0
    weighted_price: float = 0
    i = 0

    while total_volumes < self.volume and i + 1 < len(volumes):
      _volume_taken = min(self.volume - total_volumes, volumes[i + 1])
      total_volumes += _volume_taken
      weighted_price += prices[i] * (_volume_taken * 1.0 / self.volume)
      i += 1
    return weighted_price

class Lipton(InstantMetric):
  def bidask_imbalance(self, snapshot: Snapshot):
    q_b = snapshot.bid_volumes[0]
    q_a = snapshot.ask_volumes[0]
    imbalance = (q_b - q_a) / (q_b + q_a)
    return MetricData('bidask-imbalance', snapshot.symbol, imbalance)

  def upward_probability(self, snapshot: Snapshot, p_xy=-0.5): # todo: should I consider depth or only best prices available
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
