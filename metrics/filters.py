from dataloader.utils.data import Snapshot
from typing import Dict
import numpy as np


class Filters:

  class Filter:
    def filter(self, snapshot: Snapshot):
      pass

  class LevelFilter:

    def __init__(self, levels: int):
      self.levels: int = levels
      self.snapshots: Dict[str, Snapshot] = {}

      self.price_idx: np.array = np.arange(0, levels + 1, step=2)
      self.volume_idx: np.array = self.price_idx + 1


    def _store_levels(self, snapshot: Snapshot):
      self.stored_bid_levels_price = snapshot.bids[self.price_idx]
      self.stored_ask_levels_price = snapshot.asks[self.price_idx]
      self.stored_bid_levels_volume = snapshot.bids[self.volume_idx]
      self.stored_ask_levels_volume = snapshot.asks[self.volume_idx]

    def filter(self, snapshot: Snapshot) -> bool: # todo: write tests
      symbol: str = snapshot.market
      stored_snapshot: Snapshot = self.snapshots.get(symbol, None)
      
      if stored_snapshot is None:
        self.snapshots[symbol] = snapshot
        self._store_levels(snapshot)
        return True
      else:
        bid_levels_price = stored_snapshot.bids[self.price_idx]

        if bid_levels_price != self.stored_bid_levels_price:
          self._store_levels(snapshot)
          return True

        ask_levels_price = stored_snapshot.asks[self.price_idx]
        if ask_levels_price != self.stored_ask_levels_price:
          return True

        bid_levels_volume = stored_snapshot.bids[self.volume_idx]
        if bid_levels_volume != self.stored_bid_levels_volume:
          self._store_levels(snapshot)
          return True

        ask_levels_volume = stored_snapshot.asks[self.volume_idx]
        if ask_levels_volume != self.stored_ask_levels_volume:
          self._store_levels(snapshot)
          return True

        return False
