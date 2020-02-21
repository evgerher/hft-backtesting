import datetime
from dataclasses import dataclass
from typing import *
from dataloader.callbacks.connectors import Connector

from dataloader.callbacks.message import TradeMessage, MetaMessage
from utils import logger
import numpy as np

@dataclass
class Snapshot: # todo: may be sort on construct ?
  # todo: it is asserted that bids and asks are sorted
  market: str
  timestamp: datetime.datetime.timestamp
  bid_prices: np.array
  bid_volumes: np.array
  ask_prices: np.array
  ask_volumes: np.array

  volume_indices = np.arange(1, 50, 2)
  price_indices = np.arange(0, 50, 2)

  @staticmethod
  def from_sides(timestamp: datetime.datetime.timestamp, market: str, bids: np.array, asks: np.array) -> 'Snapshot':
    a_p, a_v = Snapshot.sort_side(asks, False)
    b_p, b_v = Snapshot.sort_side(bids, True)
    return Snapshot(market, timestamp, b_p, b_v, a_p, a_v)

  @staticmethod
  def sort_side(side: np.array, is_bid=False): # todo: test it
    """

    :param side: array of 50 elements: 25 pairs of (price, volume)
    :param is_ask: flag of ask side, if so -> reverse order
    :return:
    """
    # prices: np.array = side[Snapshot.price_indices]
    price_idxs = Snapshot.price_indices
    price_idxs = np.array(sorted(price_idxs, key = lambda idx: side[idx], reverse=is_bid))
    prices = side[price_idxs]
    volumes = side[price_idxs + 1]
    return prices, volumes.astype(np.int)

  def __str__(self):
    return f'Snapshot :: market={self.market}, ' \
           f'highest bid price,volume = ({self.bid_prices[0], self.bid_volumes[0]}), ' \
           f'lowest ask price, volume = ({self.ask_prices[0], self.ask_volumes[0]})'

class SnapshotBuilder:
  def __init__(self, market: str, state: List[Dict]):
    self.market = market
    self.mapping = {}
    self.free = []
    self.data: List = [0] * 100

    sells, buys = 0, 50 # even - price, odd - size
    for s in state:
      if s['side'] in 'Sell':  # asks
        self.mapping[s['id']] = sells
        self.data[sells] = float(s['price'])
        self.data[sells+1] = s['size']
        sells += 2
      else:  # bids
        self.mapping[s['id']] = buys
        self.data[buys] = float(s['price'])
        self.data[buys + 1] = s['size']
        buys += 2

    # state=[{'id': 8799192250, 'side': 'Sell', 'size': 59553, 'price': 8077.5}, ...], market=XBTUSD

  def apply(self, delta: list, action: str):
    if action in 'update':
      for update in delta:
        try:
          self.data[self.mapping[update['id']] + 1] = update['size']
        except Exception as e:
          print(e)
          raise e
    elif action in 'insert': # [{"id": 8799193300, "side": "Sell", "size": 491901}, {"id": 8799193450, "side": "Sell", "size": 1505581}]
      for insert in delta:
        _id = insert['id']
        idx: int = self.free.pop(0)
        self.mapping[_id] = idx
        self.data[idx] = float(insert['price'])
        self.data[idx + 1] = insert['size']
    elif action in 'delete': # [{"id":29699996493,"side":"Sell"},{"id":29699996518,"side":"Buy"}]}
      for delete in delta:
        _id = delete['id']
        idx: int = self.mapping[_id]
        self.data[idx + 1] = 0
        self.free.append(idx)
        del self.mapping[_id]

  def to_store(self) -> (str, datetime.datetime.timestamp, list):
    return (self.market, datetime.datetime.utcnow(), self.data)

  def to_snapshot(self) -> 'Snapshot':
    asks = np.array(self.data[0:50])
    bids = np.array(self.data[50:])
    # todo: dislike for datetime now()
    return Snapshot.from_sides(self.market, datetime.datetime.now(), bids, asks)

  def __str__(self):
    bid = max([self.data[x] for x in range(50, 100, 2)])
    ask = min([self.data[x] for x in range(0, 50, 2)])
    return f'Snapshot :: market={self.market}, highest bid = {bid}, lowest ask = {ask}'


class Data_Preprocessor:
  def __init__(self, connector: Connector):
    self.connector = connector
    self.snapshots: Dict[str, SnapshotBuilder] = {}
    self.counter = 0

  def _preprocess_partial(self, partial: dict) -> list:
    pass

  def _preprocess_update(self, tick: dict) -> list:
    pass

  def _get_message_meta(self, msg: Dict[str, str]) -> 'MetaMessage':
    pass

  def callback(self, msg: dict):
    meta = self._get_message_meta(msg)
    if meta.action is None:
      return
    elif meta.table in 'trade':
      trade: TradeMessage = TradeMessage.unwrap_data(msg)
      if '.' in trade.symbol:
        self.connector.store_index(trade)
      else:
        self.connector.store_trade(trade)
      return
    else: # process snapshot action
      if meta.action in 'partial':
        state = self._preprocess_partial(msg)
        snapshot: SnapshotBuilder = SnapshotBuilder(meta.symbol, state)
        self.snapshots[meta.symbol] = snapshot
      else:
        update = self._preprocess_update(msg)
        snapshot: SnapshotBuilder = self.snapshots[meta.symbol]
        snapshot.apply(update, meta.action)

      self.counter += 1

      self.connector.store_snapshot(*snapshot.to_store())
      if self.counter % 1000 == 0:
        logging.info(f"Inserted 1.000 more: {self.snapshots}")
        self.counter = 0


class Bitmex_Data(Data_Preprocessor):
  def _preprocess_partial(self, partial: dict) -> list:
    return self.__preprocess_dict(partial)

  def _preprocess_update(self, update: dict) -> list:
    return self.__preprocess_dict(update)

  def __preprocess_dict(self, tick: dict) -> list:
    data = []
    for x in tick['data']:
      del x['symbol']
      data.append(x)
    return data

  def _get_message_meta(self, msg: Dict[str, str]) -> 'MetaMessage':
    table = msg.get('table', None)
    action = msg.get('action', None)
    if action is None:
      return MetaMessage(None, None, None)
    return MetaMessage(table, action, msg['data'][0]['symbol'])