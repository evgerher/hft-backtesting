import datetime

from hft.dataloader.callbacks.clickhouse import clickhouse_cmds
from hft.dataloader import Connector
from clickhouse_driver import Client
from hft.dataloader import TradeMessage
from hft.utils.data import OrderBook
from hft.utils.logger import setup_logger

logger = setup_logger("<clickhouse>", "INFO")


class ClickHouse(Connector):
  def create_client(self):
    logger.info("Reestablish connection")
    return Client(self.db_host, password=self.db_pwd)

  def __init__(self, db_host=None, db_pwd=None):
    self.db_host = db_host if db_host is not None else 'localhost'
    self.db_pwd = db_pwd if db_pwd is not None else ''

    client = self.create_client()
    # client.execute(clickhouse_cmds.create_snapshots)
    client.execute(clickhouse_cmds.create_orderbook10)
    client.execute(clickhouse_cmds.create_indexes)
    client.execute(clickhouse_cmds.create_trades)

    self.client = client
    self.total_snapshots = 0
    self.snapshot_counter = 0
    self.orderbook_counter = 0
    self.trades_counter = 0

  def store_trade(self, trade: TradeMessage):
    logger.debug(f"Insert trade: symbol={trade.symbol} {trade.size} pieces for {trade.price}, "
                 f"action={trade.action} on side={trade.side} @ {trade.timestamp}")


    # trade.action=partial means current values (when application is starting)
    self.client.execute('insert into trades_orderbook_10_03_20 values', [
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
      logger.info('Trade ping clickhouse')
      self.client.connection.ping()

  def store_snapshot(self, symbol: str, timestamp: datetime.datetime, data: list):
    logger.debug(f"Insert snapshot: {timestamp}, symbol={symbol}")
    self.client.execute('insert into snapshots values', [[timestamp, timestamp.microsecond  // 1000, symbol] + data])
    self.snapshot_counter += 1

    if self.snapshot_counter % 2500 == 0:
      self.client.connection.ping()

    if self.snapshot_counter % 5000 == 0:
      logger.info('Snapshot reconnect to clickhouse')
      self.client.disconnect()
      self.client = self.create_client()

  def store_orderbook(self, orderbook: OrderBook):
    logger.debug(f"Insert orderbook: {orderbook.timestamp}, symbol={orderbook.symbol}")

    ap = orderbook.ask_prices.tolist()
    av = orderbook.ask_volumes.tolist()
    bp = orderbook.bid_prices.tolist()
    bv = orderbook.bid_volumes.tolist()
    self.client.execute('insert into orderbook_10_03_20 values',
                        [[orderbook.timestamp,
                          orderbook.timestamp.microsecond  // 1000,
                          orderbook.symbol] + ap + av + bp + bv])
    self.orderbook_counter += 1

    if self.orderbook_counter % 2500 == 0:
      self.client.connection.ping()

    if self.orderbook_counter % 5000 == 0:
      self.client.disconnect()
      self.client = self.create_client()

  def store_index(self, trade: TradeMessage):
    logger.info(f"Insert index: {trade.timestamp}")
    self.total_snapshots += 1
    self.client.execute('insert into indexes_10_03_20 values', [
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
