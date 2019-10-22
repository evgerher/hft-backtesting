from pyq import q, K # requires $QHOME to be defined
import os
import datetime

class Snapshot:
  def __init__(self, market: str, state: list):
    self.market = market
    self.data = {x['id']: x for x in state}
    # state=[{'id': 8799192250, 'side': 'Sell', 'size': 59553, 'price': 8077.5}, ...], market=XBTUSD
    # todo: keep only 50

  def apply(self, delta: list, action: str):
    if action in 'update':
      for update in delta:
        self.data[update['id']]['size'] = update['size']
    elif action in 'insert': # [{"id": 8799193300, "side": "Sell", "size": 491901}, {"id": 8799193450, "side": "Sell", "size": 1505581}]
      self.data.update({x['id']: x for x in delta})
    elif action in 'delete': # [{"id":29699996493,"side":"Sell"},{"id":29699996518,"side":"Buy"}]}
      for delete in delta:
        del self.data[delete['id']]

  def to_dict(self) -> dict:
    # todo: transform here into insert statement
    return {'market': self.market, 'state': self.data.values()}

  def __str__(self):
    # todo: differentiate buys and sells
    N = len(self.data)
    return f'market={self.market}, state_length={N}'

class Index:
  @staticmethod
  def unwrap_data(d: dict) -> (str, str, float):
    data = d['data'][-1]
    symbol = data['symbol']
    timestamp: str = data['timestamp'].replace('-', '.')
    price = data['price']
    return symbol, timestamp, price
    # '.BETHXBT', '2019.10.21T23:20:00.000Z', 0.02121


class KDB_Connector:

  def __init__(self):
    self._tables = {}

    # self.index_table = q('index_table:([] symbol:`symbol$(); timestamp:`timestamp$(); price: `float$())')
    # q.call('snapshot_table:([] ')
    self.h = q.hopen(':localhost:12000')

    self.snapshot_counter = 0
    self.index_counter    = 0

  def generate_csv_file(self, name):
    dt = datetime.datetime.now()
    return f'{name}-{dt.month}.{dt.day}:{dt.hour}:{dt.minute}.csv'

  def store(self, snapshot: dict):
    self.snapshot_counter += 1
    print(f'{self.counter}: Stored in KBT')
    assert len(snapshot['state']) == 50
    # q.insert()

    if self.snapshot_counter == 2:
      self.reload('snapshot_table')

  def store_index(self, symbol: str, timestamp: str, price: float):
    self.index_counter += 1

    # 'd:([] symbol:`symbol$(); timestamp:`timestamp$(); price: `float$())'
    # 'upsert[`d; (`.BETHXBT; `timestamp$(2019.10.21T23:20:00.000Z); 0.02121)]'
    msg = f'upsert[`index_table; (`{symbol}; `timestamp$({timestamp}); {price})]'
    print(msg)
    self.h(('insert_index', symbol, datetime.datetime.strptime(timestamp, "%Y.%m.%dT%H:%M:%S.%fZ"), price))
    if self.index_counter == 2:
      self.reload('index_table')

  def reload(self, table: str):
    q(f'save `:{table}.csv')
    csv_file = self.generate_csv_file(table)
    os.rename(f'{table}.csv', csv_file)


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
      self.connector.store_index(symbol, timestamp, price)
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

      self.connector.store(snapshot.to_dict())

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
