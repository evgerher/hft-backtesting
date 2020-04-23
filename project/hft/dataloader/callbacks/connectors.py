import datetime
from hft.dataloader import TradeMessage
from hft.utils.data import OrderBook
from abc import ABC, abstractmethod


class Connector(ABC):
  @abstractmethod
  def store_snapshot(self, market: str, timestamp:datetime.datetime, data: list):
    raise NotImplementedError

  @abstractmethod
  def store_index(self, trade: TradeMessage):
    raise NotImplementedError

  @abstractmethod
  def store_trade(self, trade: TradeMessage):
    raise NotImplementedError

  @abstractmethod
  def store_orderbook(self, orderbook: OrderBook):
    raise NotImplementedError
