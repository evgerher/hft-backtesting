import datetime
from dataloader.callbacks.message import TradeMessage
from utils.data import OrderBook


class Connector:
  def store_snapshot(self, market: str, timestamp:datetime.datetime, data: list):
    pass

  def store_index(self, trade: TradeMessage):
    pass

  def store_trade(self, trade: TradeMessage):
    pass

  def store_orderbook(self, orderbook: OrderBook):
    pass
