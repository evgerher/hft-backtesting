import datetime
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from hft.utils.logger import setup_logger

logger = setup_logger('<data>', 'INFO')


@dataclass
class Trade:
  symbol: str
  timestamp: datetime.datetime
  price: float
  volume: int
  side: int
  __slots__ = ['symbol', 'timestamp', 'price', 'volume', 'side', '__weakref__']

  def __str__(self):
    return f'<trade, symbol={self.symbol}, timestamp:{self.timestamp}, side={self.side}, price={self.price}, volume={self.volume}>'

  def to_list(self):
    return [self.symbol, self.timestamp, self.side, self.price, self.volume]

  def __hash__(self):
    return hash(self.timestamp)

  def label(self) -> str:
    return f'{self.symbol} {self.side}'

@dataclass
class OrderBook:
  symbol: str
  timestamp: datetime.datetime
  bid_prices: np.array
  bid_volumes: np.array
  ask_prices: np.array
  ask_volumes: np.array
  __slots__ = ['symbol', 'timestamp', 'bid_prices', 'bid_volumes', 'ask_prices', 'ask_volumes', '__weakref__']

  @staticmethod
  def from_bitmex_orderbook(msg: dict) -> List['OrderBook']:
    orderbooks = []
    for data in msg['data']:
      timestamp = datetime.datetime.strptime(data['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
      symbol = data['symbol']
      asks_pv = np.array(data['asks'])
      bids_pv = np.array(data['bids'])

      ap = asks_pv[:,0]
      av = asks_pv[:,1].astype(np.int)
      bp = bids_pv[:,0]
      bv = bids_pv[:,1].astype(np.int)

      orderbooks.append(OrderBook(symbol, timestamp, bp, bv, ap, av))
    return orderbooks

  @staticmethod
  def from_sides(timestamp: datetime.datetime, symbol: str, bids: np.array, asks: np.array, depth:int=25) -> 'OrderBook':
    # assert depth % 2 == 0
    a_p, a_v = OrderBook.sort_side(asks, False)
    b_p, b_v = OrderBook.sort_side(bids, True)
    return OrderBook(symbol, timestamp, b_p[:depth], b_v[:depth], a_p[:depth], a_v[:depth])

  @staticmethod
  def sort_side(side: np.array, is_bid=False):
    """

    :param side: array of 50 elements: 25 pairs of (price, volume_total)
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
           f'best bid,volume_total=({self.bid_prices[0], self.bid_volumes[0]}), ' \
           f'lowest ask,volume_total=({self.ask_prices[0], self.ask_volumes[0]})>'


@dataclass
class DeltaValue:
  timestamp: datetime.datetime
  value: int
  __slots__ = ['timestamp', 'value', '__weakref__']

@dataclass
class Delta:
  timestamp: datetime.datetime
  symbol: str
  quote_side: int
  diff: np.array
  __slots__ = ['timestamp', 'symbol', 'quote_side', 'diff', '__weakref__']
