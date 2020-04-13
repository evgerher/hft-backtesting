import datetime
from dataclasses import dataclass
from typing import Tuple

from hft.utils.consts import Statuses, QuoteSides

initial_id = 0

@dataclass
class OrderStatus:
  id: int
  status: int
  at: datetime.datetime
  volume_total: int
  volume: int


  def __str__(self):
    return f'<order-status id={self.id}, status={self.status}, at={self.at}>'

  @staticmethod
  def finish(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, Statuses.FINISHED, timestamp, -1, -1)

  @staticmethod
  def cancel(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, Statuses.CANCEL, timestamp, -1, -1)

  @staticmethod
  def partial(id: int, timestamp: datetime.datetime, volume_total: int, volume) -> 'OrderStatus':
    return OrderStatus(id, Statuses.PARTIAL, timestamp, volume_total, volume)


@dataclass
class OrderRequest:
  id: int
  command: int
  price: float
  volume: int
  symbol: str
  side: int
  created: datetime.datetime
  volume_filled: int

  def label(self) -> Tuple[str, int, float]:
    return (self.symbol, self.side, self.price)

  def __str__(self):
    return f'<order-request id={self.id}, command={self.command}, symbol={self.symbol}, side={self.side}, ' \
           f'volume={self.volume}, price={self.price}>'

  @staticmethod
  def _generate_id() -> int:
    global initial_id
    id = initial_id
    initial_id += 1
    return id

  @staticmethod
  def cancelOrder(id: int) -> 'OrderRequest':
    # todo: return money if cancelled
    return OrderRequest(id, Statuses.CANCEL, None, None, None, None, None, 0)

  @staticmethod
  def create(price: float, volume: int, symbol: str, side:int, timestamp: datetime.datetime, command=Statuses.NEW) -> 'OrderRequest':
    id = OrderRequest._generate_id()
    return OrderRequest(id, command, price, volume, symbol, side, timestamp, 0)

  @staticmethod
  def create_ask(price: float, volume: int, symbol: str, timestamp: datetime.datetime) -> 'OrderRequest':
    return OrderRequest.create(price, volume, symbol, QuoteSides.ASK, timestamp, Statuses.NEW)

  @staticmethod
  def create_bid(price: float, volume: int, symbol: str, timestamp: datetime.datetime) -> 'OrderRequest':
    return OrderRequest.create(price, volume, symbol, QuoteSides.BID, timestamp, Statuses.NEW)
