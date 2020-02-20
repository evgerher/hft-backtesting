from dataloader.utils.data import Snapshot
from typing import Dict


class Filters:

  class Filter:
    def filter(self, snapshot: Snapshot):
      pass

  class LevelFilter:

    def __init__(self, levels: int):
      self.levels: int = levels
      self.snapshots: Dict[str, Snapshot] = {}

    def _store_levels(self, snapshot: Snapshot):
      self.stored_bid_levels_price = snapshot.bid_prices[self.levels]
      self.stored_ask_levels_price = snapshot.ask_prices[self.levels]
      self.stored_bid_levels_volume = snapshot.bid_volumes[self.levels]
      self.stored_ask_levels_volume = snapshot.ask_volumes[self.levels]

    def filter(self, snapshot: Snapshot) -> bool: # todo: write tests
      symbol: str = snapshot.market
      stored_snapshot: Snapshot = self.snapshots.get(symbol, None)
      
      if stored_snapshot is None:
        self.snapshots[symbol] = snapshot
        self._store_levels(snapshot)
        return True
      else:
        bid_levels_price = stored_snapshot.bid_prices[self.levels]

        if bid_levels_price != self.stored_bid_levels_price:
          self._store_levels(snapshot)
          return True

        ask_levels_price = stored_snapshot.ask_prices[self.levels]
        if ask_levels_price != self.stored_ask_levels_price:
          return True

        bid_levels_volume = stored_snapshot.bid_volumes[self.levels]
        if bid_levels_volume != self.stored_bid_levels_volume:
          self._store_levels(snapshot)
          return True

        ask_levels_volume = stored_snapshot.ask_volumes[self.levels]
        if ask_levels_volume != self.stored_ask_levels_volume:
          self._store_levels(snapshot)
          return True

        return False
