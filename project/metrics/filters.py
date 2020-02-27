from utils.data import Snapshot
from typing import Dict, List
from utils.logger import  setup_logger
import numpy as np

logger = setup_logger('<filter>', 'DEBUG')


class Filters:

  class Filter:
    def filter(self, snapshot: Snapshot):
      pass

  class DepthFilter(Filter):

    def __init__(self, levels: int):
      self.levels: int = levels
      self.snapshots: Dict[str, Snapshot] = {}
      self.stored_bid_levels_price : Dict[str, List] = {}
      self.stored_ask_levels_price : Dict[str, List] = {}
      self.stored_bid_levels_volume: Dict[str, List] = {}
      self.stored_ask_levels_volume: Dict[str, List] = {}

    def __str__(self):
      return f'<Depth filter for n={self.levels}>'

    def _store_levels(self, snapshot: Snapshot):
      self.stored_bid_levels_price[snapshot.market] = snapshot.bid_prices[:self.levels]
      self.stored_ask_levels_price[snapshot.market] = snapshot.ask_prices[:self.levels]
      self.stored_bid_levels_volume[snapshot.market] = snapshot.bid_volumes[:self.levels]
      self.stored_ask_levels_volume[snapshot.market] = snapshot.ask_volumes[:self.levels]

    def filter(self, snapshot: Snapshot) -> bool:
      symbol: str = snapshot.market
      stored_snapshot: Snapshot = self.snapshots.get(symbol, None)
      
      if stored_snapshot is None:
        self.snapshots[symbol] = snapshot
        self._store_levels(snapshot)
        return True
      else:
        bid_levels_price = snapshot.bid_prices[:self.levels]
        blp = bid_levels_price != self.stored_bid_levels_price[snapshot.market]
        if (blp).any():
          logger.debug(f'Bid level price altered, on depth={np.where(blp == True)[0]}')
          self._store_levels(snapshot)
          return True

        ask_levels_price = snapshot.ask_prices[:self.levels]
        alp = ask_levels_price != self.stored_ask_levels_price[snapshot.market]
        if (alp).any():
          logger.debug(f'Ask level price altered, on depth={np.where(alp == True)[0]}')
          self._store_levels(snapshot)
          return True

        bid_levels_volume = snapshot.bid_volumes[:self.levels]
        blv = bid_levels_volume != self.stored_bid_levels_volume[snapshot.market]
        if (blv).any():
          logger.debug(f'Bid level volume altered, on depth={np.where(blv == True)[0]}')
          self._store_levels(snapshot)
          return True

        ask_levels_volume = snapshot.ask_volumes[:self.levels]
        alv = ask_levels_volume != self.stored_ask_levels_volume[snapshot.market]
        if (alv).any():
          logger.debug(f'Ask level volume altered, on depth={np.where(alv == True)[0]}')
          self._store_levels(snapshot)
          return True

        return False
