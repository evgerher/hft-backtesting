from utils.consts import QuoteSides
from utils.types import Delta
from utils.data import OrderBook
from typing import Dict, List, Optional
from utils.logger import  setup_logger
import numpy as np
from abc import ABC, abstractmethod

logger = setup_logger('<filter>', 'INFO')

class Filters:

  class Filter(ABC):

    @abstractmethod
    def process(self, snapshot: OrderBook) -> Optional:
      raise NotImplementedError

  class SymbolFilter(Filter):
    def __init__(self, symbol):
      self.symbol = symbol

    def filter(self, snapshot: OrderBook) -> Optional:
      if snapshot.symbol == self.symbol:
        return True
      return None


  class DepthFilter(Filter):

    def __init__(self, levels: int):
      self.levels: int = levels
      self.stored_bid_price : Dict[str, List] = {}
      self.stored_ask_price : Dict[str, List] = {}
      self.stored_bid_volume: Dict[str, List] = {}
      self.stored_ask_volume: Dict[str, List] = {}

    def __str__(self):
      return f'<Depth filter for n={self.levels}>'

    def _store_levels(self, snapshot: OrderBook):
      self.stored_bid_price[snapshot.symbol] = snapshot.bid_prices
      self.stored_ask_price[snapshot.symbol] = snapshot.ask_prices
      self.stored_bid_volume[snapshot.symbol] = snapshot.bid_volumes
      self.stored_ask_volume[snapshot.symbol] = snapshot.ask_volumes

    def __delta_level_added(self, price_new, price_old, volume_new, volume_old) -> np.array:
      return np.stack((price_new[0], volume_new[0]))

    def __delta_level_consumed(self, price_new, price_old, volume_new, volume_old) -> np.array:
      # in most cases it will be 1, but multiple levels may be consumed
      shift = np.where(price_old == price_new[0])[0] # get single item
      if shift.size > 0:
        shift = shift[0]
        volume_delta = volume_new[0] - volume_old[shift]
        volume_delta = np.concatenate((-volume_old[:shift], [volume_delta]))
        prices = price_old[:shift + 1]
        selector = volume_delta != 0
        return np.stack((prices[selector], volume_delta[selector]))
      else:
        logger.critical(f'Critical prices: {price_new}; {price_old}')
        return np.stack((price_old, -volume_old))

    def process(self, snapshot: OrderBook) -> Optional[Delta]:
      stored_memory = self.stored_bid_price.get(snapshot.symbol, None)
      
      if stored_memory is None:
        self._store_levels(snapshot)
        return (snapshot.timestamp, snapshot.symbol, QuoteSides.INIT, np.empty((0,)))
      else:
        bid_price = snapshot.bid_prices[0]
        blp = bid_price != self.stored_bid_price[snapshot.symbol][0]
        if blp:
          if bid_price > self.stored_bid_price[snapshot.symbol][0]: # new level is added
            logger.debug(f'Bid price increased, level=0')
            answer = np.stack(([snapshot.bid_prices[0]], [snapshot.bid_volumes[0]]))
          else: # level is eaten
            logger.debug(f'Bid price descreased, level=0')
            answer = self.__delta_level_consumed(snapshot.bid_prices, self.stored_bid_price[snapshot.symbol],
                                                 snapshot.bid_volumes, self.stored_bid_volume[snapshot.symbol])
          result = (snapshot.timestamp, snapshot.symbol, QuoteSides.BID_ALTER, answer)
          self._store_levels(snapshot)
          return result

        ask_price = snapshot.ask_prices[0]
        alp = ask_price != self.stored_ask_price[snapshot.symbol][0]
        if alp:
          if ask_price > self.stored_ask_price[snapshot.symbol][0]: # level is consumed
            logger.debug(f'Ask price increased, level=0')
            answer = self.__delta_level_consumed(snapshot.ask_prices, self.stored_ask_price[snapshot.symbol],
                                                 snapshot.ask_volumes, self.stored_ask_volume[snapshot.symbol])
          else: # new level is added
            logger.debug(f'Ask price descreased, level=0')
            answer = np.stack(([snapshot.ask_prices[0]], [snapshot.ask_volumes[0]]))
          result = (snapshot.timestamp, snapshot.symbol, QuoteSides.ASK_ALTER, answer)
          self._store_levels(snapshot)
          return result

        bid_levels_volume = snapshot.bid_volumes[:self.levels]
        blv = bid_levels_volume - self.stored_bid_volume[snapshot.symbol][:self.levels]
        selector = blv != 0
        if selector.any():
          logger.debug(f'Bid volume altered, level={np.where(blv == True)[0]}')
          self._store_levels(snapshot)
          return (snapshot.timestamp, snapshot.symbol, QuoteSides.BID, np.stack((snapshot.bid_prices[:self.levels][selector], blv[selector])))

        ask_volume = snapshot.ask_volumes[:self.levels]
        alv = ask_volume - self.stored_ask_volume[snapshot.symbol][:self.levels]
        selector = alv != 0
        if selector.any():
          logger.debug(f'Ask volume altered, level={np.where(alv == True)[0]}')
          self._store_levels(snapshot)
          return (snapshot.timestamp, snapshot.symbol, QuoteSides.ASK, np.stack((snapshot.ask_prices[:self.levels][selector], alv[selector])))

      return None
