import datetime
from typing import Optional, List, Union, Tuple

from hft.utils import helper
from hft.utils.data import OrderBook, Trade
from hft.utils.helper import fix_timestamp, fix_trades
from hft.utils.logger import setup_logger
import pandas as pd
from abc import ABC, abstractmethod

logger = setup_logger('<reader>', 'INFO')

class Reader(ABC):

  def __init__(self, moment: datetime.datetime):
    self.initial_moment = moment
    self.current_timestamp = moment

  class Row:
    pass

  def __iter__(self):
    return self

  @abstractmethod
  def __next__(self):
    raise NotImplementedError

class ListReader(Reader):
  def __init__(self, items: List[Union[OrderBook, Trade]]):
    self.items = items
    moment = items[0].timestamp
    super().__init__(moment)
    self.idx = 0

  def __next__(self) -> Union[Trade, OrderBook]:
    if self.idx == len(self.items):
      raise StopIteration

    item = self.items[self.idx]
    self.idx += 1
    return item

  def __getitem__(self, idx):
    return self.items[idx]

class SnapshotReader(Reader):

  def __init__(self, snapshot_file: str, trades_file: Optional[str] = None, nrows: int = 10000, stop_after: int = None, depth_to_load:int=10):
    """
    :param snapshot_file: to read
    :param trades_file: to read
    :param nrows: 10000 - number of rows to read per update
    :param stop_after: stop iteration over dataset after amount of iterations
    """
    self._snapshot_file = snapshot_file
    self._trades_file = trades_file

    self._snapshot_idx, self._trades_idx = 0, 0
    self._total_snapshots, self._total_trades = 0, 0

    self._nrows = nrows
    self._pairs_to_load =  depth_to_load
    self._snapshots_df: pd.DataFrame = self.__read_csv(self._snapshot_file)
    self._snapshots_df = fix_timestamp(self._snapshots_df, 0, 1)

    self.__limit_snapshot = len(self._snapshots_df)
    self._snapshot = self._load_snapshot()

    self._finished_trades = self._trades_file is None
    if self._trades_file is not None:
      self._trades_df: pd.DataFrame = self.__read_csv(self._trades_file)
      self._trades_df = fix_trades(self._trades_df, 1, 2)

      self._limit_trades = len(self._trades_df)
      self._trade = self.__load_trade()
      initial_trade = self._trades_df.iloc[0, 1]
    else:
      initial_trade = None
      self._limit_trades = 0

    self._read_first_trades = True
    self.stop_after = stop_after

    initial_snapshot = self._snapshots_df.iloc[0, 0]
    initial_trade = initial_trade or initial_snapshot
    super().__init__(min(initial_trade, initial_snapshot))

  def __read_csv(self, fname, skiprows=0):
    return pd.read_csv(fname, header=None, sep=',',
                       quotechar='"', error_bad_lines=False,
                       skiprows=skiprows, nrows=self._nrows)

  def __update_df(self, file, skiprows=0):
    df = self.__read_csv(file, skiprows)
    limit = len(df)
    return df, limit

  def __next__(self) -> Tuple[Union[Trade, OrderBook], bool]:  # snapshot or trade

    # end condition
    if (self.__limit_snapshot != self._nrows and self._snapshot_idx == self.__limit_snapshot) or \
        (self.stop_after is not None and self._total_snapshots + self._snapshot_idx == self.stop_after + 1):
      self._total_snapshots += self._snapshot_idx
      logger.debug(f"Finished snapshot_file {self._snapshot_file}, read {self._total_snapshots} rows")
      raise StopIteration

    if self._limit_trades != 0 and self._read_first_trades or (
        not self._finished_trades
        and self._trade.timestamp <= self._snapshot.timestamp):
      trade = self._trade
      self._trade = self.__load_trade()

      if self._read_first_trades and self._trade.timestamp >= self._snapshot.timestamp:
        self._read_first_trades = False

      obj = (trade, False)
      self.current_timestamp = trade.timestamp
    else:
      snapshot = self._snapshot
      self._snapshot = self._load_snapshot()
      obj = (snapshot, True)
      self.current_timestamp = snapshot.timestamp

    # checks on files' reloads
    if self._snapshot_idx >= self._nrows:
      self._total_snapshots += self._snapshot_idx
      logger.info(f"Reload snapshot file: total-snapshots={self._total_snapshots}")
      self._snapshots_df, self.__limit_snapshot = self.__update_df(self._snapshot_file, self._total_snapshots)
      self._snapshots_df = fix_timestamp(self._snapshots_df, 0, 1)
      self._snapshot_idx = 0

    if self._trades_file is not None and (self._limit_trades != self._nrows and self._trades_idx == self._limit_trades):
      self._total_trades += self._trades_idx
      logger.critical(f"Finished trades_file {self._trades_file}, read {self._total_trades} rows")
      self._finished_trades = True

    if self._trades_idx >= self._nrows:
      self._total_trades += self._trades_idx
      logger.info(f"Reload trades file: total-trades={self._total_trades}")
      self._trades_df, self._limit_trades = self.__update_df(self._trades_file, self._total_trades)
      self._trades_df = fix_trades(self._trades_df, 1, 2)
      self._trades_idx = 0

    return obj

  def __load_trade(self) -> Trade:
    if not self._finished_trades:
      row: pd.Series = self._trades_df.iloc[self._trades_idx, :]
      self._trades_idx += 1
      return Trade(row['symbol'], row['timestamp'], row['side'], row['price'], row['volume'])
    return self._trade

  def _load_snapshot(self) -> OrderBook:
    row: pd.Series = self._snapshots_df.iloc[self._snapshot_idx, :]
    timestamp, market, bids, asks = helper.snapshot_line_parser(row)
    self._snapshot_idx += 1

    return OrderBook.from_sides(timestamp, market, bids, asks, self._pairs_to_load)


  def __str__(self):
    return f'<snapshot-reader on snapshot_file={self._snapshot_file}, ' \
           f'trades_file={self._trades_file}, ' \
           f'batch_nrows={self._nrows}>'


class OrderbookReader(SnapshotReader):
  def _load_snapshot(self) -> OrderBook:
    row: pd.Series = self._snapshots_df.iloc[self._snapshot_idx, :]
    self._snapshot_idx += 1
    return helper.orderbook_line_parse(row, self._pairs_to_load)

  def __str__(self):
    return f'<orderbook-reader on orderbook_file={self._snapshot_file}, ' \
           f'trades_file={self._trades_file}, ' \
           f'batch_nrows={self._nrows}>'
