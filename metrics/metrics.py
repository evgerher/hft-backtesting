from utils.data import Snapshot
import numpy as np


class Metrics:
  @staticmethod
  def VWAP_bid(snapshot: Snapshot) -> float:
    return Metrics._VWAP(snapshot.bids)

  @staticmethod
  def VWAP_ask(snapshot: Snapshot) -> float:
    return Metrics._VWAP(snapshot.asks)

  @staticmethod
  def _VWAP(items: np.array) -> float:
    volumes = np.sum(items[Snapshot.volume_indices])
    return items[Snapshot.price_indices] * items[Snapshot.volume_indices] / volumes

  @staticmethod
  def VWAP_midpoint(vwap_bid: float, vwap_ask: float) -> float:
    return (vwap_bid + vwap_ask) / 2

  @staticmethod
  def bidask_imbalance(snapshot: Snapshot):
    q_b = snapshot.bids[snapshot.best_bid_volume_index()]
    q_a = snapshot.asks[snapshot.best_ask_volume_index()]
    return (q_b - q_a) / (q_b + q_a)

  @staticmethod
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
