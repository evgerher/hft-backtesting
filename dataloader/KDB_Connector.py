import time
import datetime
from pyq import q
import os
import logging

class KDB_Connector:

  def _load_init(self, fname='init.q'):
    with open('init.q', 'r') as f:
      return f.readlines()

  def _initialize(self):
    # self.h = q.hopen(':localhost:12000')
    for q_cmd in self._load_init():
      q(q_cmd)
      logging.info(f"Initialized {q_cmd}")

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

      if self.snapshot_counter > 100:
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
    q(f'append[`snapshot_table;{msg}]')
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
