import datetime
from dataclasses import dataclass
from typing import List


@dataclass
class MetaMessage:
  table: str
  action: str
  symbol: str


@dataclass
class TradeMessage:
  symbol: str
  timestamp: datetime.datetime
  price: float
  size: int
  action: str
  side: str

  # {"table": "trade", "action": "insert", "data": [
  #   {"timestamp": "2020-02-04T22:08:32.518Z", "symbol": "XBTUSD", "side": "Buy", "size": 100, "price": 9135.5,
  #    "tickDirection": "PlusTick", "trdMatchID": "9a286908-520d-ed91-c7a0-f32ed8abfc43", "grossValue": 1094600,
  #    "homeNotional": 0.010946, "foreignNotional": 100}]}
  # symbol: str, timestamp: str, price: float, size: int, action: str, side: str
  @staticmethod
  def unwrap_data(d: dict) -> List['TradeMessage']:
    trades = []
    for data in d['data']:
      symbol = data['symbol']
      timestamp = datetime.datetime.strptime(data['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
      price = data['price']
      size = data['size']
      action = d['action']
      side = data['side']
      trades.append(TradeMessage(symbol, timestamp, price, size, action, side))
    return trades
    # '.BETHXBT', '2019.10.21T23:20:00.000Z', 0.02121, 100, 'insert', 'Buy'
