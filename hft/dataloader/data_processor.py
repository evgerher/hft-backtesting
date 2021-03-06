from typing import List, Dict
import datetime
import numpy as np

from hft.dataloader.callbacks.connectors import Connector
from hft.utils.data import OrderBook
from hft.utils.logger import setup_logger
from hft.dataloader.callbacks.message import TradeMessage, MetaMessage
from abc import ABC, abstractmethod

logger = setup_logger('<data-loader>', 'INFO')

class SnapshotBuilder:
  def __init__(self, symbol: str, state: List[Dict]):
    self.symbol = symbol
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

    # state=[{'id': 8799192250, 'side': 'Sell', 'size': 59553, 'price': 8077.5}, ...], symbol=XBTUSD

  def apply(self, updates: list, action: str):
    if action in 'update':
      for update in updates:
        try:
          self.data[self.mapping[update['id']] + 1] = update['size']
        except Exception as e:
          print(e)
          raise e
    elif action in 'insert': # [{"id": 8799193300, "side": "Sell", "size": 491901}, {"id": 8799193450, "side": "Sell", "size": 1505581}]
      for insert in updates:
        _id = insert['id']
        idx: int = self.free.pop(0)
        self.mapping[_id] = idx
        self.data[idx] = float(insert['price'])
        self.data[idx + 1] = insert['size']
    elif action in 'delete': # [{"id":29699996493,"side":"Sell"},{"id":29699996518,"side":"Buy"}]}
      for delete in updates:
        _id = delete['id']
        idx: int = self.mapping[_id]
        self.data[idx + 1] = 0
        self.free.append(idx)
        del self.mapping[_id]

  def to_store(self) -> (str, datetime.datetime, list):
    return (self.symbol, datetime.datetime.utcnow(), self.data)

  def to_snapshot(self) -> 'OrderBook':
    asks = np.array(self.data[0:50])
    bids = np.array(self.data[50:])
    return OrderBook.from_sides(datetime.datetime.utcnow(), self.symbol, bids, asks)

  def __str__(self):
    bid = max([self.data[x] for x in range(50, 100, 2)])
    ask = min([self.data[x] for x in range(0, 50, 2)])
    return f'<Snapshot :: symbol={self.symbol}, highest bid = {bid}, lowest ask = {ask}>'


class Data_Preprocessor(ABC):
  def __init__(self, connector: Connector):
    self.connector = connector
    self.snapshots: Dict[str, SnapshotBuilder] = {}
    self.counter = 0

  @abstractmethod
  def _preprocess_partial(self, partial: dict) -> list:
    raise NotImplementedError

  @abstractmethod
  def _preprocess_update(self, tick: dict) -> list:
    raise NotImplementedError

  @abstractmethod
  def _get_message_meta(self, msg: Dict[str, str]) -> 'MetaMessage':
    raise NotImplementedError

  def callback(self, msg: dict):
    meta = self._get_message_meta(msg)
    if meta.action is None:
      return
    elif meta.table in 'trade':
      trades: List[TradeMessage] = TradeMessage.unwrap_data(msg)
      for trade in trades:
        if '.' in trades[-1].symbol:
          self.connector.store_index(trade)
        else:
          self.connector.store_trade(trade)
      return
    elif 'orderBook' in meta.table:
      orderbooks: List[OrderBook] = OrderBook.from_bitmex_orderbook(msg)
      for orderbook in orderbooks:
        self.connector.store_orderbook(orderbook)
    else: # process snapshot action
      if meta.action == 'partial':
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
        logger.info(f"Inserted 1.000 more: {self.snapshots}")
        self.counter = 0


class Bitmex_Data(Data_Preprocessor):

  def __str__(self):
    return f'<bitmex_data preprocessor>'

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
