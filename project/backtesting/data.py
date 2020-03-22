import datetime
from dataclasses import dataclass
from typing import Tuple

initial_id = 0


@dataclass
class OrderStatus:
  id: int
  status: str
  at: datetime.datetime
  volume: int

  def __str__(self):
    return f'<order-status id:{self.id} status:{self.status} volume:{self.volume} at:{self.at}>'

  @staticmethod
  def finish(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, 'finished', timestamp, -1)

  @staticmethod
  def remove(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, 'removed', timestamp, -1)

  @staticmethod
  def partial(id: int, timestamp: datetime.datetime, volume: int) -> 'OrderStatus':
    return OrderStatus(id, 'partial', timestamp, volume)


@dataclass
class OrderRequest:
  id: int
  command: str
  price: float
  volume: int
  symbol: str
  side: str
  created: datetime.datetime
  volume_filled: int

  def label(self) -> Tuple[str, str, float]:
    return (self.symbol, self.side, self.price)

  def __str__(self):
    return f'<order-request id:{self.id} command:{self.command} symbol:{self.symbol} side:{self.side}>'

  @staticmethod
  def _generate_id() -> int:
    global initial_id
    id = initial_id
    initial_id += 1
    return id

  @staticmethod
  def cancelOrder(id: int) -> 'OrderRequest':
    # todo: return money if cancelled
    return OrderRequest(id, 'delete', None, None, None, None, None, 0)

  @staticmethod
  def create(price: float, volume: int, symbol: str, side:str, timestamp: datetime.datetime, command='new') -> 'OrderRequest':
    id = OrderRequest._generate_id()
    return OrderRequest(id, command, price, volume, symbol, side, timestamp, 0)

  @staticmethod
  def create_ask(price: float, volume: int, symbol: str, timestamp: datetime.datetime) -> 'OrderRequest':
    return OrderRequest.create(price, volume, symbol, 'ask', timestamp, 'new')

  @staticmethod
  def create_bid(price: float, volume: int, symbol: str, timestamp: datetime.datetime) -> 'OrderRequest':
    return OrderRequest.create(price, volume, symbol, 'bid', timestamp, 'new')
