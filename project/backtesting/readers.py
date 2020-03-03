import datetime
from typing import Optional

from utils import helper
from utils.data import OrderBook, Trade
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger('<reader>', 'INFO')

class Reader:

  def __init__(self, moment: datetime.datetime):
    self.initial_moment = moment

  class Row:
    pass

  def __iter__(self):
    return self

  def __next__(self):
    pass

class SnapshotReader(Reader):

  def __init__(self, snapshot_file: str, trades_file: Optional[str] = None, nrows: int = 10000, stop_after: int = None, pairs_to_load:int=10):
    """
    :param snapshot_file: to read
    :param trades_file: to read
    :param nrows: 10000 - number of rows to read per update
    :param stop_after: stop iteration over dataset after amount of iterations
    """
    self._snapshot_file = snapshot_file
    self._trades_file = trades_file

    self.__snapshot_idx, self.__trades_idx = 0, 0
    self._total_snapshots, self._total_trades = 0, 0

    self._nrows = nrows
    self._pairs_to_load =  pairs_to_load
    self.__snapshots_df: pd.DataFrame = self.__read_csv(self._snapshot_file)
    self.__limit_snapshot = len(self.__snapshots_df)
    self.__snapshot = self.__load_snapshot()


    if self._trades_file is not None:
      self.__trades_df: pd.DataFrame = self.__read_csv(self._trades_file)
      self.__limit_trades = len(self.__trades_df)
      self.__trade = self.__load_trade()


    self.__stop_after = stop_after

    super().__init__(helper.convert_to_datetime(self.__trades_df.iloc[0, 1]))

  def __read_csv(self, fname, skiprows=0):
    return pd.read_csv(fname, header=None, sep=',',
                       quotechar='"', error_bad_lines=False,
                       skiprows=skiprows, nrows=self._nrows)

  def __update_df(self, file, skiprows=0):
    df = self.__read_csv(file, skiprows)
    limit = len(df)
    return df, limit

  def __next__(self):  # snapshot or trade

    # end condition
    if (self.__limit_snapshot != self._nrows and self.__snapshot_idx == self.__limit_snapshot) or \
        (self.__stop_after is not None and self.__snapshot_idx == self.__stop_after):
      self._total_snapshots += self.__snapshot_idx
      logger.debug(f"Finished snapshot_file {self._snapshot_file}, read {self._total_snapshots} rows")
      raise StopIteration

    # checks on files' reloads
    if self.__snapshot_idx + 1 >= self._nrows:
      self._total_snapshots += self.__snapshot_idx
      self.__snapshots_df, self.__limit_snapshot = self.__update_df(self._snapshot_file, self.__snapshot_idx)
      self.__snapshot_idx = 0

    if self.__trades_idx + 1 >= self._nrows:
      self._total_trades += self.__trades_idx
      self.__trades_df, self.__limit_trades = self.__update_df(self._trades_file, self.__trades_idx)
      self.__trades_idx = 0


    # select whom to return

    if self._trades_file is not None and self.__trade.timestamp < self.__snapshot.timestamp:
      trade = self.__trade
      self.__trade = self.__load_trade()
      return trade
    else:
      snapshot = self.__snapshot
      self.__snapshot = self.__load_snapshot()
      return snapshot

  def __load_trade(self) -> Trade:
    row: pd.Series = self.__trades_df.iloc[self.__trades_idx, :]
    self.__trades_idx += 1
    return helper.trade_line_parser(row)

  def __load_snapshot(self) -> OrderBook:
    row: pd.Series = self.__snapshots_df.iloc[self.__snapshot_idx, :]
    timestamp, market, bids, asks = helper.snapshot_line_parser(row)
    self.__snapshot_idx += 1

    return OrderBook.from_sides(timestamp, market, bids, asks, self._pairs_to_load)


  def __str__(self):
    return f'<Snapshot reader on snapshot_file={self._snapshot_file}, ' \
           f'trades_file={self._trades_file}, ' \
           f'batch_nrows={self._nrows}>'
