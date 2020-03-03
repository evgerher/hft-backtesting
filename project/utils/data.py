import datetime
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from utils.logger import setup_logger

logger = setup_logger('<data>', 'DEBUG')


@dataclass
class Trade:
  symbol: str
  timestamp: datetime.datetime
  side: str
  price: float
  volume: int

  def label(self) -> str:
    return f'{self.symbol} {self.side}'

  def belongs_to(self, initial: 'Snapshot', target: 'Snapshot', levels:int=3, seconds_limit=15):
    result: Tuple[str, str, np.ndarray] = target.diff(initial, levels)
    snapshot_side, pv, diff = result
    snapshot_price = initial.ask_prices[0] if snapshot_side == 'ask' else initial.bid_prices[0]
    volume_total = np.sum([diff[0]])

    trade_side = 'bid' if self.side == 'Sell' else 'ask'

    # if snapshot_side in 'ask':
    #   pass
    # elif snapshot_side in 'bid':
    #   pass

    return snapshot_side == trade_side \
        and volume_total == self.volume \
        and snapshot_price == self.price \
        and (self.timestamp - initial.timestamp).seconds < seconds_limit




@dataclass
class Snapshot:
  symbol: str
  timestamp: datetime.datetime
  bid_prices: np.array
  bid_volumes: np.array
  ask_prices: np.array
  ask_volumes: np.array

  def diff(self, other: 'Snapshot', levels:int=3) -> Tuple[str, str, np.ndarray]:
    bid_levels_price = other.bid_prices[:levels]
    blp = bid_levels_price - self.bid_prices[:levels]
    if (blp != 0).any():
      logger.debug(f'Bid level price altered, on depth={np.where(blp == True)[0]}')
      return ('bid', 'price', blp)

    ask_levels_price = other.ask_prices[:levels]
    alp = ask_levels_price - self.ask_prices[:levels]
    if (alp != 0).any():
      logger.debug(f'Ask level price altered, on depth={np.where(alp != 0)[0]}')
      return ('ask', 'price', alp)

    bid_levels_volume = other.bid_volumes[:levels]
    blv = bid_levels_volume - self.bid_volumes[:levels]
    if (blv != 0).any():
      logger.debug(f'Bid level volume altered, on depth={np.where(blv != 0)[0]}')
      return ('bid', 'volume', blv)

    ask_levels_volume = other.ask_volumes[:levels]
    alv = ask_levels_volume - self.ask_volumes[:levels]
    if (alv != 0).any():
      logger.debug(f'Ask level volume altered, on depth={np.where(alv != 0)[0]}')
      return ('ask' ,'volume', alv)

    return ('equal', '', np.empty(levels, np.float))

  def __sub__(self, other):
    return other.diff(self)

  @staticmethod
  def from_sides(timestamp: datetime.datetime, symbol: str, bids: np.array, asks: np.array, depth:int=25) -> 'Snapshot':
    # assert depth % 2 == 0
    a_p, a_v = Snapshot.sort_side(asks, False)
    b_p, b_v = Snapshot.sort_side(bids, True)
    return Snapshot(symbol, timestamp, b_p[:depth], b_v[:depth], a_p[:depth], a_v[:depth])

  @staticmethod
  def sort_side(side: np.array, is_bid=False):
    """

    :param side: array of 50 elements: 25 pairs of (price, volume)
    :param is_ask: flag of ask side, if so -> reverse order
    :return:
    """
    # prices: np.array = side[Snapshot.price_indices]
    price_idxs = np.arange(0, len(side), 2)
    price_idxs = np.array(sorted(price_idxs, key = lambda idx: side[idx], reverse=is_bid))
    prices = side[price_idxs]
    volumes = side[price_idxs + 1]
    return prices, volumes.astype(np.int)

  def __str__(self):
    return f'<Snapshot :: symbol={self.symbol}, ' \
           f'highest bid price,volume = ({self.bid_prices[0], self.bid_volumes[0]}), ' \
           f'lowest ask price, volume = ({self.ask_prices[0], self.ask_volumes[0]})>'
