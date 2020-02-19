from dataclasses import dataclass
from typing import List

from dataloader.utils.data import Snapshot
import numpy as np

@dataclass
class Metric:
  name: str
  value: float

class Metrics:

  class MetricsEvaluator:
    def evaluate(self, Snapshot) -> List['Metric']:
      pass

  class _VWAP(MetricsEvaluator):

    def evaluate(self, snapshot: Snapshot):
      vwap_bid, vwap_ask = self.VWAP_bid(snapshot), self.VWAP_ask(snapshot)
      midpoint = self.VWAP_midpoint(vwap_bid, vwap_ask)
      return (
        Metric(f'{self.__str__()} bid', vwap_bid),
        Metric(f'{self.__str__()} ask', vwap_ask),
        Metric(f'{self.__str__()} midpoint', midpoint)
      )

    def _evaluate_side(self, items: np.array) -> float:
      pass

    def VWAP_bid(self, snapshot: Snapshot) -> float:
      return self._evaluate_side(snapshot.bids)

    def VWAP_ask(self, snapshot: Snapshot) -> float:
      return self._evaluate_side(snapshot.asks)

    def VWAP_midpoint(self, vwap_bid: float, vwap_ask: float) -> float:
      return (vwap_bid + vwap_ask) / 2

  class VWAP_depth(_VWAP):
    def __str__(self):
      return f'VWAP (Depth): {self.levels}'

    def __init__(self, levels = 3):
      self.levels = levels
      self.volume_indices = Snapshot.volume_indices[:self.levels]
      self.price_indices = Snapshot.price_indices[:self.levels]

    def _evaluate_side(self, items: np.array) -> float:
      # volumes are assumed to be sorted
      volumes = np.sum(items[self.volume_indices])
      return items[self.price_indices] * items[self.volume_indices] / volumes

  class VWAP_volume(_VWAP):

    def __str__(self):
      return f'VWAP (Volume): {self.volume}'

    def __init__(self, volume: int = 1e6):
      self.volume = volume

    def _evaluate_side(self, items: np.array) -> float:
      total_volumes = 0
      pairs: dict = {}
      i = 0
      while total_volumes < self.volume:
        _volume_taken = min(self.volume - total_volumes, items[i + 1])
        total_volumes += _volume_taken
        pairs[items[i]] = _volume_taken / self.volume
        i += 1
      return sum(list(map(lambda p: p[0] * p[1], pairs.items())))

  def bidask_imbalance(snapshot: Snapshot):
    q_b = snapshot.bids[snapshot.best_bid_volume_index()]
    q_a = snapshot.asks[snapshot.best_ask_volume_index()]
    return (q_b - q_a) / (q_b + q_a)

  def lipton_upward_probability(snapshot: Snapshot, p_xy=-0.5): # todo: should I consider depth or only best prices available
    """
    x - bid quote sizes
    y - ask quote sizes
    :param snapshot:
    :param p_xy: correlation between the depletion and replenishment of the bid and ask queues' diffusion processes (typically negative)
    :return:
    """
    # todo: how to evaluate p_xy ?
    # todo: implement p_xy
    x = snapshot.bids[snapshot.best_bid_volume_index()]
    y = snapshot.asks[snapshot.best_ask_volume_index()]
    sqrt_corr = np.sqrt((1 + p_xy) / (1 - p_xy))
    return 1. / 2 * (1. - np.arctan(sqrt_corr * (y - x) / (y + x)) / np.arctan(sqrt_corr))
