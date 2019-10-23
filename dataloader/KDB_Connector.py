import threading
import time
import datetime
from pyq import q
import os
import utils

logger = utils.setup_logger('KDB.KDB_Connector')


class KDB_Connector(threading.Thread):

  def _load_init(self, fname='init.q'):
    with open('init.q', 'r') as f:
      return f.readlines()


  def _initialize(self):
    # self.h = q.hopen(':localhost:12000')
    for q_cmd in self._load_init():
      q(q_cmd)
      logger.info(f"Initialized {q_cmd}")

    self._snapshot_counter = 0
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
        self.store_snapshot(*snapshot)

      for _ in range(indexes_count - 1):
        index = self.indexes.pop(0)
        self.store_index(*index)

      time.sleep(1.)

  def generate_csv_file(self, name):
    dt = datetime.datetime.now()
    return f'{name}-{dt.month}.{dt.day}:{dt.hour}:{dt.minute}.csv'

  def store_snapshot(self, market, data, timestamp):
    # 1 - timestamp; 2 - symbol; 3-102 - snapshot
    # 3-52: (price size) pairs Sell
    # 53-102: (price size) pairs Buy
    q(f'`snapshot_table upsert {tuple([timestamp, market] + data)}')
    # self.h(tuple(['insert_snapshot', timestamp, market] + data))
    logger.info(f'{self._snapshot_counter}: Stored in KBT')
    self._snapshot_counter += 1

  def store_index(self, symbol: str, timestamp: str, price: float):
    self._index_counter += 1

    # 'd:([] symbol:`symbol$(); timestamp:`timestamp$(); price: `float$())'
    # 'upsert[`d; (`.BETHXBT; `timestamp$(2019.10.21T23:20:00.000Z); 0.02121)]'
    logger.info(f'upsert[`index_table; (`{symbol}; `timestamp$({timestamp}); {price})]')
    q(f'`index_table upsert {tuple(symbol, datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"), price)}')
    # self.h(('insert_index', symbol, datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"), price))
    # if self._index_counter == 2:
    #   self.reload('index_table')

  def reload(self, table: str):
    q(f'save `:{table}.csv')
    csv_file = self.generate_csv_file(table)
    os.rename(f'{table}.csv', csv_file)