import datetime

from dataloader.callbacks.clickhouse import clickhouse_cmds
from dataloader.callbacks.connectors import Connector
import logging
from clickhouse_driver import Client
from dataloader.callbacks.message import TradeMessage


class ClickHouse(Connector):
  def create_client(self):
    logging.info("Reestablish connection")
    return Client(self.db_host, password=self.db_pwd)

  def __init__(self, db_host=None, db_pwd=None):
    self.db_host = db_host if db_host is not None else 'localhost'
    self.db_pwd = db_pwd if db_pwd is not None else ''

    client = self.create_client()
    client.execute(clickhouse_cmds.create_snapshots)
    client.execute(clickhouse_cmds.create_indexes)
    client.execute(clickhouse_cmds.create_trades)

    self.client = client
    self.total_snapshots = 0
    self.snapshot_counter = 0
    self.trades_counter = 0

  def store_trade(self, trade: TradeMessage):
    logging.info(f"Insert trade: symbol={trade.symbol} {trade.size} pieces for {trade.price}, "
                 f"action={trade.action} on side={trade.side} @ {trade.timestamp}")
    self.client.execute('insert into trades values', [
      (
        trade.symbol,
        trade.timestamp,
        trade.timestamp.microsecond // 1000,
        trade.price,
        trade.size,
        trade.action,
        trade.side
      )])
    self.trades_counter += 1

    if self.trades_counter % 2500 == 0:
      self.client.connection.ping()

  def store_snapshot(self, market, timestamp: datetime.datetime.timestamp, data: list):
    logging.info(f"Insert snapshot: {timestamp}, market={market}")
    self.client.execute('insert into snapshots values', [[timestamp, timestamp.microsecond  // 1000, market] + data])
    self.snapshot_counter += 1

    if self.snapshot_counter % 2500 == 0:
      self.client.connection.ping()

    if self.snapshot_counter % 5000 == 0:
      self.client.disconnect()
      self.client = self.create_client()

  def store_index(self, trade: TradeMessage):
    logging.info(f"Insert index: {trade.timestamp}")
    self.total_snapshots += 1
    self.client.execute('insert into indexes values', [
      (trade.symbol, trade.timestamp, trade.price)
    ])

    if self.total_snapshots % 200 == 0:
      self.client.disconnect()
      self.client = self.create_client()
      self.client.connection.ping()

  def save_csv(self):
    snaps = self.client.execute('select * from snapshots limit 5000')
    import pandas as pd
    snaps: pd.DataFrame = pd.DataFrame(snaps)
    snaps.to_csv(path_or_buf='snapshots.csv', header=False, index=False)


if __name__ == '__main__':
  client = ClickHouse()
  client.save_csv()