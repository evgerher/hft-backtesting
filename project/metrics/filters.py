from utils.data import OrderBook
from typing import Dict, List
from utils.logger import  setup_logger
import numpy as np

logger = setup_logger('<filter>', 'INFO')


class Filters:

  class Filter:
    def filter(self, snapshot: OrderBook):
      pass

  class DepthFilter(Filter):

    def __init__(self, levels: int):
      self.levels: int = levels
      self.snapshots: Dict[str, OrderBook] = {}
      self.stored_bid_levels_price : Dict[str, List] = {}
      self.stored_ask_levels_price : Dict[str, List] = {}
      self.stored_bid_levels_volume: Dict[str, List] = {}
      self.stored_ask_levels_volume: Dict[str, List] = {}

    def __str__(self):
      return f'<Depth filter for n={self.levels}>'

    def _store_levels(self, snapshot: OrderBook):
      self.stored_bid_levels_price[snapshot.symbol] = snapshot.bid_prices[:self.levels]
      self.stored_ask_levels_price[snapshot.symbol] = snapshot.ask_prices[:self.levels]
      self.stored_bid_levels_volume[snapshot.symbol] = snapshot.bid_volumes[:self.levels]
      self.stored_ask_levels_volume[snapshot.symbol] = snapshot.ask_volumes[:self.levels]

    def filter(self, snapshot: OrderBook) -> bool:
      symbol: str = snapshot.symbol
      stored_snapshot: OrderBook = self.snapshots.get(symbol, None)
      
      if stored_snapshot is None:
        self.snapshots[symbol] = snapshot
        self._store_levels(snapshot)
        return True
      else:
        bid_levels_price = snapshot.bid_prices[0]
        blp = bid_levels_price != self.stored_bid_levels_price[snapshot.symbol][0]
        if blp:
          logger.debug(f'Bid level price altered, on level=0')
          self._store_levels(snapshot)
          return True

        ask_levels_price = snapshot.ask_prices[0]
        alp = ask_levels_price != self.stored_ask_levels_price[snapshot.symbol][0]
        if alp:
          logger.debug(f'Ask level price altered, on level=0')
          self._store_levels(snapshot)
          return True

        bid_levels_volume = snapshot.bid_volumes[:self.levels]
        blv = bid_levels_volume != self.stored_bid_levels_volume[snapshot.symbol]
        if (blv).any():
          logger.debug(f'Bid level volume altered, on level={np.where(blv == True)[0]}')
          self._store_levels(snapshot)
          return True

        ask_levels_volume = snapshot.ask_volumes[:self.levels]
        alv = ask_levels_volume != self.stored_ask_levels_volume[snapshot.symbol]
        if (alv).any():
          logger.debug(f'Ask level volume altered, on level={np.where(alv == True)[0]}')
          self._store_levels(snapshot)
          return True

        return False
