import time
import datetime
import os
import logging
from clickhouse_driver import Client

class Connector:

  def store_snapshot(self, market, data, timestamp:datetime.datetime.timestamp):
    pass

  def store_index(self, symbol: str, timestamp: str, price: float):
    pass

class ClickHouse(Connector):
  def __init__(self):
    client = Client('localhost')
    client.execute('CREATE TABLE IF NOT EXISTS snapshots (moment TIMESTAMP, symbol FixedString(10), '
                   'x0 Float32,x1 UInt32,x2 Float32,x3 UInt32,x4 Float32,x5 UInt32,x6 Float32,x7 UInt32,x8 Float32,x9 UInt32,x10 Float32,x11 UInt32,x12 Float32,x13 UInt32,x14 Float32,x15 UInt32,x16 Float32,x17 UInt32,x18 Float32,x19 UInt32,x20 Float32,x21 UInt32,x22 Float32,x23 UInt32,x24 Float32,x25 UInt32,x26 Float32,x27 UInt32,x28 Float32,x29 UInt32,x30 Float32,x31 UInt32,x32 Float32,x33 UInt32,x34 Float32,x35 UInt32,x36 Float32,x37 UInt32,x38 Float32,x39 UInt32,x40 Float32,x41 UInt32,x42 Float32,x43 UInt32,x44 Float32,x45 UInt32,x46 Float32,x47 UInt32,x48 Float32,x49 UInt32,x50 Float32,x51 UInt32,x52 Float32,x53 UInt32,x54 Float32,x55 UInt32,x56 Float32,x57 UInt32,x58 Float32,x59 UInt32,x60 Float32,x61 UInt32,x62 Float32,x63 UInt32,x64 Float32,x65 UInt32,x66 Float32,x67 UInt32,x68 Float32,x69 UInt32,x70 Float32,x71 UInt32,x72 Float32,x73 UInt32,x74 Float32,x75 UInt32,x76 Float32,x77 UInt32,x78 Float32,x79 UInt32,x80 Float32,x81 UInt32,x82 Float32,x83 UInt32,x84 Float32,x85 UInt32,x86 Float32,x87 UInt32,x88 Float32,x89 UInt32,x90 Float32,x91 UInt32,x92 Float32,x93 UInt32,x94 Float32,x95 UInt32,x96 Float32,x97 UInt32,x98 Float32,x99 UInt32) '
                   'ENGINE = MergeTree() '
                   'ORDER BY moment')
    client.execute('CREATE TABLE IF NOT EXISTS indexes (symbol FixedString(15), moment TIMESTAMP, price Float32) '
                   'ENGINE = MergeTree()'
                   'ORDER BY moment')

    self.client = client
    self.total_snapshots = 0
    self.snapshot_counter = 0

  def store_snapshot(self, market, data, timestamp:datetime.datetime.timestamp):
    self.client.execute('insert into snapshots values', [[timestamp, market] + data])

  def store_index(self, symbol: str, timestamp: str, price: float):
    self.client.execute('insert into indexes values', [(symbol, datetime.datetime.strptime(timestamp, "%Y.%m.%dT%H:%M:%S.%f"), price)])

class KDB_Connector(Connector):

  def _load_init(self, fname='init.q'):
    with open('init.q', 'r') as f:
      return f.readlines()

  def _initialize(self):
    # self.h = q.hopen(':localhost:12000')
    # for q_cmd in self._load_init():
    #   q(q_cmd)
    #   logging.info(f"Initialized {q_cmd}")
    q('\l init.q')
    self.total_snapshots = 0
    self.snapshot_counter = 0
    self._index_counter = 0
    self.finished = False
    self.snapshots = []
    self.indexes = []

  def run(self):
    self._initialize()
    time.sleep(3)
    while not self.finished:
      snapshots_count = len(self.snapshots)
      indexes_count = len(self.indexes)
      for _ in range(snapshots_count - 3):
        snapshot = self.snapshots.pop(0)
        self._store_snapshot(*snapshot)

      for _ in range(indexes_count - 1):
        index = self.indexes.pop(0)
        self._store_index(*index)

      logging.info(f'KDB_Connector :: stored {self.total_snapshots + self.snapshot_counter}')
      time.sleep(3.)

      if self.snapshot_counter > 50:
        q('.u.end[]')
        # self._reload('snapshot_table')
        # q('.Q.gc[]')
        self.total_snapshots += self.snapshot_counter
        self.snapshot_counter = 0

      if self._index_counter > 1000:
        # self._reload('index_table')
        # q('.Q.gc[]')
        self._index_counter = 0

  def generate_csv_file(self, name):
    dt = datetime.datetime.now()
    return f'{name}-{dt.day}.{dt.month}.{dt.year}:{dt.hour}:{dt.minute}:{dt.second}.csv'

  def _store_snapshot(self, market, data, timestamp:datetime.datetime.timestamp):
    # 1 - timestamp; 2 - symbol; 3-102 - snapshot
    # 3-52: (price size) pairs Sell
    # 53-102: (price size) pairs Buy

    msg = f'({timestamp.strftime("%Y.%m.%dD%H:%M:%S.%f")}; `{market}; {str(tuple(data)).replace(",", ";")[1:-1]})'

    # q(f'`snapshot_table upsert {msg}')
    q(f'upd[`snapshot_table;{msg}]')
    # self.h(tuple(['insert_snapshot', timestamp, market] + data))
    # logging.debug(f'{self.snapshot_counter}: Stored in KDB')
    self.snapshot_counter += 1

  def _store_index(self, symbol: str, timestamp: str, price: float):
    self._index_counter += 1

    # 'd:([] symbol:`symbol$(); timestamp:`timestamp$(); price: `float$())'
    # 'upsert[`d; (`.BETHXBT; `timestamp$(2019.10.21T23:20:00.000Z); 0.02121)]'
    msg = f'`{symbol}; `timestamp${timestamp}; {price}'
    # logging.debug(f'`index_table upsert ({msg})')
    q(f'`index_table upsert ({msg})')
    # self.h(('insert_index', symbol, datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"), price))
    # if self._index_counter == 2:
    #   self.reload('index_table')

  def _reload(self, table: str):
    logging.info(f"Reloaded table {table}")
    q(f'save `:{table}.csv')
    csv_file = self.generate_csv_file(table)
    os.rename(f'{table}.csv', csv_file)

    if table in 'snapshot':
      q('reload_snapshot[]')
    elif table in 'index':
      q('reload_index[]')

    q('.Q.gc[]')

  def close(self):
    logging.info("KDB_Connector FINISHING")
    self.finished = True
    tables = ['index_table', 'snapshot_table']
    for table in tables:
      q(f'save `:{table}.csv')
