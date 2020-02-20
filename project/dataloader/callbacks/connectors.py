import datetime
import logging
from clickhouse_driver import Client
from dataloader import config
from dataloader.callbacks.message import TradeMessage


class Connector:
  def store_snapshot(self, market, timestamp:datetime.datetime.timestamp, data: list):
    pass

  def store_index(self, trade: TradeMessage):
    pass

  def store_trade(self, trade: TradeMessage):
    pass


class ClickHouse(Connector):
  def create_client(self):
    logging.info("Reestablish connection")
    return Client(config.db_host, password=config.db_pwd)

  def __init__(self):
    client = self.create_client()
    client.execute('CREATE TABLE IF NOT EXISTS snapshots (moment TIMESTAMP, symbol FixedString(10), '
                   'x0 Float32,x1 UInt32,x2 Float32,x3 UInt32,x4 Float32,x5 UInt32,x6 Float32,x7 UInt32,x8 Float32,'
                   'x9 UInt32,x10 Float32,x11 UInt32,x12 Float32,x13 UInt32,x14 Float32,x15 UInt32,x16 Float32,'
                   'x17 UInt32,x18 Float32,x19 UInt32,x20 Float32,x21 UInt32,x22 Float32,x23 UInt32,x24 Float32,'
                   'x25 UInt32,x26 Float32,x27 UInt32,x28 Float32,x29 UInt32,x30 Float32,x31 UInt32,x32 Float32,'
                   'x33 UInt32,x34 Float32,x35 UInt32,x36 Float32,x37 UInt32,x38 Float32,x39 UInt32,x40 Float32,'
                   'x41 UInt32,x42 Float32,x43 UInt32,x44 Float32,x45 UInt32,x46 Float32,x47 UInt32,x48 Float32,'
                   'x49 UInt32,x50 Float32,x51 UInt32,x52 Float32,x53 UInt32,x54 Float32,x55 UInt32,x56 Float32,'
                   'x57 UInt32,x58 Float32,x59 UInt32,x60 Float32,x61 UInt32,x62 Float32,x63 UInt32,x64 Float32,'
                   'x65 UInt32,x66 Float32,x67 UInt32,x68 Float32,x69 UInt32,x70 Float32,x71 UInt32,x72 Float32,'
                   'x73 UInt32,x74 Float32,x75 UInt32,x76 Float32,x77 UInt32,x78 Float32,x79 UInt32,x80 Float32,'
                   'x81 UInt32,x82 Float32,x83 UInt32,x84 Float32,x85 UInt32,x86 Float32,x87 UInt32,x88 Float32,'
                   'x89 UInt32,x90 Float32,x91 UInt32,x92 Float32,x93 UInt32,x94 Float32,x95 UInt32,x96 Float32,'
                   'x97 UInt32,x98 Float32,x99 UInt32) '
                   'ENGINE=File(CSV) ')
    client.execute('CREATE TABLE IF NOT EXISTS indexes (symbol FixedString(15), moment TIMESTAMP, price Float32) '
                   'ENGINE=File(CSV)')

    client.execute(
      'CREATE TABLE IF NOT EXISTS trades (symbol FixedString(15), moment TIMESTAMP , price Float32, size INT, '
      'action FixedString(15), side FixedString(5)) '
      'ENGINE=File(CSV)')

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
    self.client.execute('insert into snapshots values', [[timestamp, market] + data])
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
