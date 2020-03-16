import datetime
from dataclasses import dataclass
from typing import Tuple

initial_id = 0


@dataclass
class OrderStatus:
  id: int
  status: str
  at: datetime.datetime

  @staticmethod
  def finish(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, 'finished', timestamp)

  @staticmethod
  def remove(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, 'removed', timestamp)


@dataclass
class OrderRequest:
  id: int
  command: str
  price: float
  volume: int
  symbol: str
  side: str
  created: datetime.datetime

  def label(self) -> Tuple[str, str, float]:
    return (self.symbol, self.side, self.price)

  @staticmethod
  def _generate_id() -> int:
    global initial_id
    id = initial_id
    initial_id += 1
    return id

  @staticmethod
  def cancelOrder(id: int) -> 'OrderRequest':
    return OrderRequest(id, 'delete', None, None, None, None, None)

  @staticmethod
  def create_ask(price: float, volume: int, symbol: str, timestamp: datetime.datetime) -> 'OrderRequest':
    id = OrderRequest._generate_id()
    return OrderRequest(id, 'new', price, volume, symbol, 'ask', timestamp)

  @staticmethod
  def create_bid(price: float, volume: int, symbol: str, timestamp: datetime.datetime) -> 'OrderRequest':
    id = OrderRequest._generate_id()
    return OrderRequest(id, 'new', price, volume, symbol, 'bid', timestamp)