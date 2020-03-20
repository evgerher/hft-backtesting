import datetime
from typing import Optional, Tuple, List, Union

from utils import helper
from utils.data import OrderBook, Trade
from utils.logger import setup_logger
import pandas as pd
from abc import ABC, abstractmethod

logger = setup_logger('<reader>', 'INFO')

class Reader(ABC):

  def __init__(self, moment: datetime.datetime):
    self.initial_moment = moment

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

  def __init__(self, snapshot_file: str, trades_file: Optional[str] = None, nrows: int = 10000, stop_after: int = None, pairs_to_load:int=10):
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
    self._pairs_to_load =  pairs_to_load
    self._snapshots_df: pd.DataFrame = self.__read_csv(self._snapshot_file)
    self.__limit_snapshot = len(self._snapshots_df)
    self._snapshot = self._load_snapshot()


    self.__finished_trades = self._trades_file is None
    if self._trades_file is not None:
      self.__trades_df: pd.DataFrame = self.__read_csv(self._trades_file)
      self._limit_trades = len(self.__trades_df)
      self._trade = self.__load_trade()
      initial_trade = helper.convert_to_datetime(self.__trades_df.iloc[0, 1])
    else:
      initial_trade = None
      self._limit_trades = 0

    self._read_first_trades = True
    self.__stop_after = stop_after

    initial_snapshot = helper.convert_to_datetime(self._snapshots_df.iloc[0, 0])
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

  def __next__(self):  # snapshot or trade

    # end condition
    if (self.__limit_snapshot != self._nrows and self._snapshot_idx == self.__limit_snapshot) or \
        (self.__stop_after is not None and self._total_snapshots + self._snapshot_idx == self.__stop_after + 1):
      self._total_snapshots += self._snapshot_idx
      logger.debug(f"Finished snapshot_file {self._snapshot_file}, read {self._total_snapshots} rows")
      raise StopIteration

    # checks on files' reloads
    if self._snapshot_idx + 1 >= self._nrows:
      self._total_snapshots += self._snapshot_idx
      self._snapshots_df, self.__limit_snapshot = self.__update_df(self._snapshot_file, self._snapshot_idx)
      self._snapshot_idx = 0

    if (self._limit_trades != self._nrows and self._trades_idx == self._limit_trades):
      self._total_trades += self._trades_idx
      logger.debug(f"Finished trades_file {self._trades_file}, read {self._total_trades} rows")
      self.__finished_trades = True

    if self._trades_idx + 1 >= self._nrows:
      self._total_trades += self._trades_idx
      self.__trades_df, self._limit_trades = self.__update_df(self._trades_file, self._trades_idx)
      self._trades_idx = 0

    # select whom to return

    if self._limit_trades != 0 and self._read_first_trades or (
        not self.__finished_trades
        and self._trade.timestamp == self._snapshot.timestamp
        and self._trade.symbol == self._snapshot.symbol): # todo: here I loose the last trade in buffer
      trade = self._trade
      self._trade = self.__load_trade()

      if self._read_first_trades and self._trade.timestamp >= self._snapshot.timestamp:
        self._read_first_trades = False

      return trade
    else:
      snapshot = self._snapshot
      self._snapshot = self._load_snapshot()
      return snapshot

  def __load_trade(self) -> Trade:
    if not self.__finished_trades:
      row: pd.Series = self.__trades_df.iloc[self._trades_idx, :]
      self._trades_idx += 1
      return helper.trade_line_parser(row)
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
