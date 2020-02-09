import datetime
from connectors import Connector
import logging

from utils import utils

logger = utils.setup_logger()

class Snapshot:
  def __init__(self, market: str, state: list):
    self.market = market
    self.mapping = {}
    self.free = []
    self.data = [0] * 100

    sells, buys = 0, 50 # even - price, odd - size
    for s in state:
      if s['side'] in 'Sell':
        self.mapping[s['id']] = sells
        self.data[sells] = float(s['price'])
        self.data[sells+1] = s['size']
        sells += 2
      else:
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
          print()
          raise e
    elif action in 'insert': # [{"id": 8799193300, "side": "Sell", "size": 491901}, {"id": 8799193450, "side": "Sell", "size": 1505581}]
      for insert in delta:
        _id = insert['id']
        idx = self.free.pop(0)
        self.mapping[_id] = idx
        self.data[idx] = float(insert['price'])
        self.data[idx + 1] = insert['size']
    elif action in 'delete': # [{"id":29699996493,"side":"Sell"},{"id":29699996518,"side":"Buy"}]}
      for delete in delta:
        _id = delete['id']
        idx = self.mapping[_id]
        self.data[idx + 1] = 0
        self.free.append(idx)
        del self.mapping[_id]

  def to_store(self) -> (str, list, datetime.datetime.timestamp):
    return (self.market, self.data, datetime.datetime.now())

  def __str__(self):
    bid = max([self.data[x] for x in range(50, 100, 2)])
    ask = min([self.data[x] for x in range(0, 50, 2)])
    return f'Snapshot :: market={self.market}, highest bid = {bid}, lowest ask = {ask}'

class TradeMessage:
  # {"table": "trade", "action": "insert", "data": [
  #   {"timestamp": "2020-02-04T22:08:32.518Z", "symbol": "XBTUSD", "side": "Buy", "size": 100, "price": 9135.5,
  #    "tickDirection": "PlusTick", "trdMatchID": "9a286908-520d-ed91-c7a0-f32ed8abfc43", "grossValue": 1094600,
  #    "homeNotional": 0.010946, "foreignNotional": 100}]}
  # symbol: str, timestamp: str, price: float, size: int, action: str, side: str
  @staticmethod
  def unwrap_data(d: dict) -> (str, str, float, int, str, str):
    data = d['data'][-1]
    symbol = data['symbol']
    timestamp: str = data['timestamp'].replace('-', '.')[:-1]
    price = data['price']
    size = data['size']
    action = d['action']
    side = data['side']
    return symbol, timestamp, price, size, action, side
    # '.BETHXBT', '2019.10.21T23:20:00.000Z', 0.02121, 100, 'insert', 'Buy'


class Data_Preprocessor:
  def __init__(self, connector: Connector):
    self.connector = connector
    self.snapshots = {}
    self.counter = 0

  def _preprocess_partial(self, partial: dict) -> list:
    pass

  def _preprocess_update(self, tick: dict) -> list:
    pass

  def _get_table_action_market(self, msg: dict) -> (str, str, str):
    pass

  def callback(self, msg: dict):
    table, action, market = self._get_table_action_market(msg)
    if action is None:
      return
    elif table in 'trade':
      symbol, timestamp, price, size, action, side = TradeMessage.unwrap_data(msg)
      if '.' in symbol:
        self.connector.store_index(symbol, timestamp, price)
      else:
        self.connector.store_trade(symbol, timestamp, price, size, action, side)
      return
    else: # process snapshot action
      if action in 'partial':
        state = self._preprocess_partial(msg)
        snapshot = Snapshot(market, state)
        self.snapshots[market] = snapshot
      else:
        update = self._preprocess_update(msg)
        snapshot = self.snapshots[market]
        snapshot.apply(update, action)

      self.counter += 1

      self.connector.store_snapshot(*snapshot.to_store())
      if self.counter > 100000 == 0:
        logging.info(f"Inserted {self.counter} more")
        self.counter = 0
      # self.connector.snapshots.append(snapshot.to_store())
      # if self.connector.snapshot_counter % 100 == 0 and self.connector.snapshot_counter != 0:
      #   logging.info(f'{self.connector.total_snapshots} :: {snapshot}')

class Bitmex_Data(Data_Preprocessor):
  def _preprocess_partial(self, partial: dict) -> list:
    return self.__preprocess_dict(partial)

  def _preprocess_update(self, update: dict) -> list:
    return self.__preprocess_dict(update)

  def __preprocess_dict(self, d: dict) -> list:
    data = []
    for x in d['data']:
      del x['symbol']
      data.append(x)
    return data

  def _get_table_action_market(self, msg: dict) -> (str, str, str):
    table = msg.get('table', None)
    action = msg.get('action', None)
    if action is None:
      return None, None, None
    return (table, action, msg['data'][0]['symbol'])
