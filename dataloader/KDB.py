import datetime
import utils
from KDB_Connector import KDB_Connector

from collections import OrderedDict

logger = utils.setup_logger()

class Snapshot:
  def __init__(self, market: str, state: list):
    self.market = market
    self.sell = OrderedDict()
    self.buy = OrderedDict()
    self.data = {'Sell': self.sell, 'Buy': self.buy}

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
    return f'market={self.market}, highest bid = {bid}, lowest ask = {ask}'

class Index:
  @staticmethod
  def unwrap_data(d: dict) -> (str, str, float):
    data = d['data'][-1]
    symbol = data['symbol']
    timestamp: str = data['timestamp'].replace('-', '.')[:-1]
    price = data['price']
    return symbol, timestamp, price
    # '.BETHXBT', '2019.10.21T23:20:00.000Z', 0.02121


class KDB_callbacker:
  def __init__(self, connector: KDB_Connector):
    self.connector = connector
    self.snapshots = {}

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

    elif table in 'trade': # Currently implemented for indexes
      symbol, timestamp, price = Index.unwrap_data(msg)
      self.connector.indexes.append((symbol, timestamp, price))
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

      self.connector.snapshots.append(snapshot.to_store())

class KDB_Bitmex(KDB_callbacker):
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

  # "action": "partial", "keys": ["symbol", "id", "side"], "types": {"symbol": "symbol", "id": "long", "side": "symbol",
  #                                                                  "size": "long", "price": "float"}, "foreignKeys": {
  #   "symbol": "instrument", "side": "side"}, "attributes": {"symbol": "parted", "id": "sorted"}, "filter": {
  #   "symbol": "XBTUSD"}, "data": [
  #   {"symbol": "XBTUSD", "id": 8799192250, "side": "Sell", "size": 59553, "price": 8077.5},
  #   {"symbol": "XBTUSD", "id": 8799192300, "side": "Sell", "size": 142712, "price": 8077},
  #   {"symbol": "XBTUSD", "id": 8799192350, "side": "Sell", "size": 54585, "price": 8076.5},
  #   {"symbol": "XBTUSD", "id": 8799192400, "side": "Sell", "size": 270514, "price": 8076},
  #   {"symbol": "XBTUSD", "id": 8799192450, "side": "Sell", "size": 61947, "price": 8075.5},
  #   {"symbol": "XBTUSD", "id": 8799192500, "side": "Sell", "size": 256530, "price": 8075},
  #   {"symbol": "XBTUSD", "id": 8799192550, "side": "Sell", "size": 48548, "price": 8074.5},
  #   {"symbol": "XBTUSD", "id": 8799192600, "side": "Sell", "size": 138474, "price": 8074},
  #   {"symbol": "XBTUSD", "id": 8799192650, "side": "Sell", "size": 78284, "price": 8073.5},
  #   {"symbol": "XBTUSD", "id": 8799192700, "side": "Sell", "size": 147337, "price": 8073},
  #   {"symbol": "XBTUSD", "id": 8799192750, "side": "Sell", "size": 142052, "price": 8072.5},
  #   {"symbol": "XBTUSD", "id": 8799192800, "side": "Sell", "size": 108037, "price": 8072},
  #   {"symbol": "XBTUSD", "id": 8799192850, "side": "Sell", "size": 341434, "price": 8071.5},
  #   {"symbol": "XBTUSD", "id": 8799192900, "side": "Sell", "size": 45055, "price": 8071},
  #   {"symbol": "XBTUSD", "id": 8799192950, "side": "Sell", "size": 137471, "price": 8070.5},
  #   {"symbol": "XBTUSD", "id": 8799193000, "side": "Sell", "size": 338218, "price": 8070},
  #   {"symbol": "XBTUSD", "id": 8799193050, "side": "Sell", "size": 152858, "price": 8069.5},
  #   {"symbol": "XBTUSD", "id": 8799193100, "side": "Sell", "size": 174383, "price": 8069},
  #   {"symbol": "XBTUSD", "id": 8799193150, "side": "Sell", "size": 61857, "price": 8068.5},
  #   {"symbol": "XBTUSD", "id": 8799193200, "side": "Sell", "size": 236125, "price": 8068},
  #   {"symbol": "XBTUSD", "id": 8799193250, "side": "Sell", "size": 80555, "price": 8067.5},
  #   {"symbol": "XBTUSD", "id": 8799193300, "side": "Sell", "size": 492701, "price": 8067},
  #   {"symbol": "XBTUSD", "id": 8799193350, "side": "Sell", "size": 181133, "price": 8066.5},
  #   {"symbol": "XBTUSD", "id": 8799193400, "side": "Sell", "size": 294605, "price": 8066},
  #   {"symbol": "XBTUSD", "id": 8799193450, "side": "Sell", "size": 1505601, "price": 8065.5},
  #   {"symbol": "XBTUSD", "id": 8799193500, "side": "Buy", "size": 1498904, "price": 8065},
  #   {"symbol": "XBTUSD", "id": 8799193550, "side": "Buy", "size": 1944, "price": 8064.5},
  #   {"symbol": "XBTUSD", "id": 8799193600, "side": "Buy", "size": 10236, "price": 8064},
  #   {"symbol": "XBTUSD", "id": 8799193650, "side": "Buy", "size": 53181, "price": 8063.5},
  #   {"symbol": "XBTUSD", "id": 8799193700, "side": "Buy", "size": 1838, "price": 8063},
  #   {"symbol": "XBTUSD", "id": 8799193750, "side": "Buy", "size": 3039, "price": 8062.5},
  #   {"symbol": "XBTUSD", "id": 8799193800, "side": "Buy", "size": 13774, "price": 8062},
  #   {"symbol": "XBTUSD", "id": 8799193850, "side": "Buy", "size": 177380, "price": 8061.5},
  #   {"symbol": "XBTUSD", "id": 8799193900, "side": "Buy", "size": 79422, "price": 8061},
  #   {"symbol": "XBTUSD", "id": 8799193950, "side": "Buy", "size": 37848, "price": 8060.5},
  #   {"symbol": "XBTUSD", "id": 8799194000, "side": "Buy", "size": 174164, "price": 8060},
  #   {"symbol": "XBTUSD", "id": 8799194050, "side": "Buy", "size": 7056, "price": 8059.5},
  #   {"symbol": "XBTUSD", "id": 8799194100, "side": "Buy", "size": 6543, "price": 8059},
  #   {"symbol": "XBTUSD", "id": 8799194150, "side": "Buy", "size": 16005, "price": 8058.5},
  #   {"symbol": "XBTUSD", "id": 8799194200, "side": "Buy", "size": 49885, "price": 8058},
  #   {"symbol": "XBTUSD", "id": 8799194250, "side": "Buy", "size": 59571, "price": 8057.5},
  #   {"symbol": "XBTUSD", "id": 8799194300, "side": "Buy", "size": 99741, "price": 8057},
  #   {"symbol": "XBTUSD", "id": 8799194350, "side": "Buy", "size": 78487, "price": 8056.5},
  #   {"symbol": "XBTUSD", "id": 8799194400, "side": "Buy", "size": 186098, "price": 8056},
  #   {"symbol": "XBTUSD", "id": 8799194450, "side": "Buy", "size": 78792, "price": 8055.5},
  #   {"symbol": "XBTUSD", "id": 8799194500, "side": "Buy", "size": 156408, "price": 8055},
  #   {"symbol": "XBTUSD", "id": 8799194550, "side": "Buy", "size": 7897, "price": 8054.5},
  #   {"symbol": "XBTUSD", "id": 8799194600, "side": "Buy", "size": 136869, "price": 8054},
  #   {"symbol": "XBTUSD", "id": 8799194650, "side": "Buy", "size": 61983, "price": 8053.5},
  #   {"symbol": "XBTUSD", "id": 8799194700, "side": "Buy", "size": 14633, "price": 8053}]}
