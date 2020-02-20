from dataclasses import dataclass
from typing import List

from dataloader.utils.data import Snapshot
import numpy as np

@dataclass
class Metric:
  name: str
  market: str
  value: float

class MetricEvaluator:
  def evaluate(self, Snapshot) -> List['Metric']:
    pass

class _VWAP(MetricEvaluator):

  def evaluate(self, snapshot: Snapshot):
    vwap_bid, vwap_ask = self.VWAP_bid(snapshot), self.VWAP_ask(snapshot)
    midpoint = self.VWAP_midpoint(vwap_bid, vwap_ask)
    return (
      Metric(f'{self.__str__()} bid', snapshot.market, vwap_bid),
      Metric(f'{self.__str__()} ask', snapshot.market, vwap_ask),
      Metric(f'{self.__str__()} midpoint', snapshot.market, midpoint)
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
    return f'VWAP (Volume): {self.volume}'

  def __init__(self, volume: int = 1e6):
    self.volume = volume

  def _evaluate_side(self, prices: np.array, volumes: np.array) -> float:
    total_volumes: int = 0
    weighted_price: float = 0
    i = 0

    while total_volumes < self.volume:
      _volume_taken = min(self.volume - total_volumes, volumes[i + 1])
      total_volumes += _volume_taken
      weighted_price += prices[i] * (_volume_taken * 1.0 / self.volume)
      i += 1
    return weighted_price

class Lipton(MetricEvaluator):
  def bidask_imbalance(self, snapshot: Snapshot):
    q_b = snapshot.bid_volumes[0]
    q_a = snapshot.ask_volumes[0]
    imbalance = (q_b - q_a) / (q_b + q_a)
    return Metric('bidask-imbalance', snapshot.market, imbalance)

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
    return Metric('Lipton-unward-probability', snapshot.market, p)
