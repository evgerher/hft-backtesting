import datetime
from dataclasses import dataclass


@dataclass
class MetaMessage:
  table: str
  action: str
  symbol: str


@dataclass
class TradeMessage:
  symbol: str
  timestamp: datetime.datetime.timestamp
  price: float
  size: int
  action: str
  side: str