from utils.data import Snapshot
import numpy as np

class Metrics:
  price_indices = np.arange(0, 50, 2)
  volume_indices = np.arange(1, 50, 2)

  def VWAP_bid(snapshot: Snapshot) -> float:
    return Metrics.VWAP(snapshot.bids)

  def VWAP_ask(snapshot: Snapshot) -> float:
    return Metrics.VWAP(snapshot.asks)

  def VWAP(items: np.array) -> float:
    volumes = np.sum(items[Metrics.volume_indices])
    return items[Metrics.price_indices] * items[Metrics.volume_indices] / volumes


  def VWAP_midpoint(vwap_bid, vwap_ask) -> float:
    return (vwap_bid + vwap_ask) / 2