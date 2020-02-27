import datetime
from dataloader.callbacks.message import TradeMessage


class Connector:
  def store_snapshot(self, market, timestamp:datetime.datetime, data: list):
    pass

  def store_index(self, trade: TradeMessage):
    pass

  def store_trade(self, trade: TradeMessage):
    pass
