import datetime
import pandas
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from utils.logger import setup_logger

logger = setup_logger('<data>', 'INFO')


@dataclass
class Trade:
  symbol: str
  timestamp: datetime.datetime
  side: str
  price: float
  volume: int

  def to_list(self):
    return [self.symbol, self.timestamp, self.side, self.price, self.volume]

  def __hash__(self):
    return hash(self.timestamp)

  def label(self) -> str:
    return f'{self.symbol} {self.side}'

  def belongs_to(self, initial: 'OrderBook', target: 'OrderBook', levels:int=3, seconds_limit=3):
    result: Tuple[str, str, np.ndarray] = target.diff(initial, levels)
    snapshot_side, pv, diff = result

    if diff[0] == 0:
      return False

    snapshot_price = initial.ask_prices[0] if snapshot_side == 'ask' else initial.bid_prices[0]
    volume_total = np.sum(diff)

    trade_side = 'bid' if self.side == 'Sell' else 'ask'
    price_condition = self.price >= snapshot_price if trade_side is 'ask' else self.price <= snapshot_price

    # if snapshot_side in 'ask':
    #   pass
    # elif snapshot_side in 'bid':
    #   pass

    return snapshot_side == trade_side \
        and np.abs(volume_total) == self.volume \
        and snapshot_price == self.price \
        and (self.timestamp - initial.timestamp).seconds < seconds_limit


@dataclass
class OrderBook:
  symbol: str
  timestamp: datetime.datetime
  bid_prices: np.array
  bid_volumes: np.array
  ask_prices: np.array
  ask_volumes: np.array

  def diff(self, other: 'OrderBook', levels:int=3) -> Tuple[str, str, np.ndarray]:
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

    logger.debug('Levels are equal')
    return ('equal', '', np.zeros(levels, np.int))

  def __sub__(self, other):
    return other.diff(self)

  @staticmethod
  def from_bitmex_orderbook(msg: dict) -> 'OrderBook':
    # 'data': [
    # {
    # 'symbol': 'XBTUSD',
    # 'bids': [[8707, 242703], [8706.5, 155], [8706, 20266], [8705.5, 112793], [8705, 16297], [8704.5, 9833], [8704, 233926], [8703.5, 403910], [8703, 462338], [8702.5, 66211]],
    # 'asks': [[8707.5, 1906350], [8708, 17855], [8708.5, 12143], [8709, 43133], [8709.5, 47247], [8710, 100078], [8710.5, 438889], [8711, 39812], [8711.5, 251992], [8712, 159765]],
    # 'timestamp': '2020-03-03T20:54:04.114Z'
    # }]}
    data = msg['data'][0]
    timestamp = datetime.datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
    symbol = data['symbol']
    asks_pv = np.array(data['asks'])
    bids_pv = np.array(data['bids'])

    ap = asks_pv[:,0]
    av = asks_pv[:,1].astype(np.int)
    bp = bids_pv[:,0]
    bv = bids_pv[:,1].astype(np.int)

    return OrderBook(symbol, timestamp, bp, bv, ap, av)

  @staticmethod
  def from_sides(timestamp: datetime.datetime, symbol: str, bids: np.array, asks: np.array, depth:int=25) -> 'OrderBook':
    # assert depth % 2 == 0
    a_p, a_v = OrderBook.sort_side(asks, False)
    b_p, b_v = OrderBook.sort_side(bids, True)
    return OrderBook(symbol, timestamp, b_p[:depth], b_v[:depth], a_p[:depth], a_v[:depth])

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
    return f'<snapshot, symbol={self.symbol}, ' \
           f'timestamp: {self.timestamp} ' \
           f'best bid,volume=({self.bid_prices[0], self.bid_volumes[0]}), ' \
           f'lowest ask,volume=({self.ask_prices[0], self.ask_volumes[0]})>'


from utils import helper

class OrderBookPandas(OrderBook):

  def __init__(self, series: pandas.Series, levels:int=5):
    self.levels = levels
    self._series = series

  def symbol(self):
    return self._series[2]

  def timestamp(self):
    return helper.convert_to_datetime(self._series[0]) + datetime.timedelta(milliseconds=int(self._series[1]))

  def ask_prices(self):
    return self._series[3:3+self.levels]

  def ask_volumes(self):
    return self._series[13:13+self.levels]

  def bid_prices(self):
    return self._series[23:23+self.levels]

  def bid_volumes(self):
    return self._series[33:33+self.levels]
