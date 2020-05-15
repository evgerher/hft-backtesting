import datetime
from typing import Optional, List, Union, Tuple, Generator

from hft.utils import helper
from hft.utils.data import OrderBook, Trade
from hft.utils.helper import fix_timestamp_drop_millis, fix_trades_rename
from hft.utils.logger import setup_logger
import pandas as pd
from abc import ABC

logger = setup_logger('<reader>', 'INFO')

class Reader(ABC):

  def __init__(self, moment: datetime.datetime):
    self.initial_moment = moment
    self.current_timestamp = moment

  def __iter__(self):
    raise NotImplementedError

class ListReader(Reader):
  def __init__(self, items: List[Union[OrderBook, Trade]]):
    self.items = items
    moment = items[0].timestamp
    super().__init__(moment)
    self.idx = 0

  def __iter__(self):
    return self

  def __next__(self) -> Union[Trade, OrderBook]:
    if self.idx == len(self.items):
      raise StopIteration()

    item = self.items[self.idx]
    self.idx += 1
    return item

  def __getitem__(self, idx):
    return self.items[idx]

class OrderbookReader(Reader):
  # todo: separate reader with and without _trades file
  # todo: move readers into separate class
  # todo: implement parallel async file read during strategy

  def __init__(self, snapshot_file: str,
               trades_file: Optional[str] = None,
               nrows: int = 300000,
               stop_after: int = None,
               depth_to_load:int=10,
               is_precomputed: bool = False):
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

    self._is_precomputed = is_precomputed
    self._nrows = nrows
    self._pairs_to_load = depth_to_load
    self._snapshots_df, self.__limit_snapshot = self._read_snapshots(self._snapshot_file, 0)
    self._snapshot_generator = self._load_snapshot()
    self._snapshot = next(self._snapshot_generator)

    self._finished_trades = self._trades_file is None
    if self._trades_file is not None:
      self._trades_df, self._limit_trades = self._read_trades(self._trades_file, 0)
      self._trade_generator = self.__load_trade()
      self._trade = next(self._trade_generator)
      initial_trade = self._trades_df.iloc[0, 1]
    else:
      initial_trade = None
      self._limit_trades = 0

    self._read_first_trades = True
    self.stop_after = stop_after

    initial_snapshot = self._snapshots_df.iloc[0, 0]
    initial_trade = initial_trade or initial_snapshot

    super().__init__(min(initial_trade, initial_snapshot))

  def _read_csv(self, fname, skiprows=0):
    return pd.read_csv(fname, header=None, sep=',',
                       quotechar='"', error_bad_lines=False,
                       skiprows=skiprows, nrows=self._nrows)

  def __update_df(self, file, skiprows=0):
    df = self._read_csv(file, skiprows)
    limit = len(df)
    return df, limit

  def __iter__(self):
    return self

  def __next__(self) -> Tuple[Union[Trade, OrderBook], bool]:  # snapshot or trade
    while True:
      if self.stop_after is not None and self._total_snapshots + self._snapshot_idx == self.stop_after + 1:
        raise StopIteration()
      if self._limit_trades != 0 and self._read_first_trades or (
          not self._finished_trades
          and self._trade.timestamp <= self._snapshot.timestamp):
        trade = self._trade
        self._trade = next(self._trade_generator)

        if self._read_first_trades and self._trade.timestamp >= self._snapshot.timestamp:
          self._read_first_trades = False

        obj = (trade, False)
        self.current_timestamp = self._trade.timestamp
      else:
        snapshot = self._snapshot
        self._snapshot = next(self._snapshot_generator)
        obj = (snapshot, True)
        self.current_timestamp = self._snapshot.timestamp

      return obj

  def try_reset(self) -> bool:
    if self._nrows is None:
      return False # finished whole file
    self._reload_snapshot_df()
    # self._trade_end_condition()
    if self._trades_file is not None:
      self._reload_trades_df()
    return self._end_condition()

  def __load_trade(self) -> Generator[Trade, None, None]:
    for row in self._trades_df.itertuples(index=False, name=None):
      self._trades_idx += 1
      yield Trade(*row)

  def _load_snapshot(self) -> Generator[OrderBook, None, None]:
    for row in self._snapshots_df.itertuples(index=False, name=None):
      self._snapshot_idx += 1
      yield helper.orderbook_line_parse(row, self._pairs_to_load)

  def __str__(self):
    return f'<orderbook-reader on orderbook_file={self._snapshot_file}, ' \
           f'_trades_file={self._trades_file}, ' \
           f'batch_nrows={self._nrows}>'

  def _read_trades(self, trades_file: str, skiprows: int):
    df, limit = self.__update_df(trades_file, skiprows)
    df = fix_trades_rename(df, 1, 2, self._is_precomputed)
    return df, limit

  def _read_snapshots(self, snapshot_file, skiprows:int):
    df, limit = self.__update_df(snapshot_file, skiprows)
    df = fix_timestamp_drop_millis(df, 0, 1, self._is_precomputed)
    return df, limit

  def _end_condition(self) -> bool:
    if (self.__limit_snapshot != self._nrows and self._snapshot_idx == self.__limit_snapshot) or \
        (self.stop_after is not None and self._total_snapshots + self._snapshot_idx == self.stop_after + 1):
      self._total_snapshots += self._snapshot_idx
      logger.debug(f"Finished snapshot_file {self._snapshot_file}, read {self._total_snapshots} rows")
      return False
    return True

  def _reload_snapshot_df(self):
    if self._snapshot_idx >= self._nrows:
      self._total_snapshots += self._snapshot_idx
      logger.info(f"Reload snapshot file: total-snapshots={self._total_snapshots}")
      self._snapshots_df, self.__limit_snapshot = self._read_snapshots(self._snapshot_file, self._total_snapshots)
      self._snapshot_idx = 0

      self._snapshot_generator = self._load_snapshot()

  def _reload_trades_df(self):
    if self._trades_idx >= self._nrows:
      self._total_trades += self._trades_idx
      logger.info(f"Reload _trades file: total-_trades={self._total_trades}")
      self._trades_df, self._limit_trades = self._read_trades(self._trades_file, self._total_trades)
      self._trades_idx = 0

      self._trade_generator = self.__load_trade()

  def _trade_end_condition(self):
    if self._trades_file is not None and (self._limit_trades != self._nrows and self._trades_idx == self._limit_trades):
      self._total_trades += self._trades_idx
      logger.critical(f"Finished _trades_file {self._trades_file}, read {self._total_trades} rows")
      self._finished_trades = True

  def total(self):
    if self.stop_after is not None:
      return self.stop_after
    return None

  def get_ending_moment(self) -> datetime.datetime:
    return min(self._trades_df.index[-1], self._snapshots_df.index[-1]).to_pydatetime()


class TimeLimitedReader(OrderbookReader):
  def __init__(self, snapshot_file: str, limit_time: str, skip_time: str = None, **kwargs): # todo: add warm-up run
    self.initial_moment = self.read_initial_moment(snapshot_file)
    limit_time: datetime.timedelta = helper.convert_to_timedelta(limit_time)
    if skip_time is not None:
      skip_time: datetime.timedelta = helper.convert_to_timedelta(skip_time)
      self.initial_moment = self.initial_moment + skip_time

    self.end_moment = self.initial_moment + limit_time
    super().__init__(snapshot_file, **kwargs)
    self._finished_snapshots = False

    if self._trades_file is not None:
      self.current_last_trade_ts = self._trades_df.iloc[-1].timestamp
    self.current_last_snapshot_ts = self._snapshots_df.iloc[-1][0]

  def _end_condition(self) -> bool:
    if self.current_timestamp >= self.end_moment or (self._finished_trades and self._finished_snapshots):
      self._total_snapshots += self._snapshot_idx
      logger.debug(f"Finished snapshot_file {self._snapshot_file}, read {self._total_snapshots} rows")
      return False
    return True

  def _reload_snapshot_df(self):
    if self.current_timestamp >= self.current_last_snapshot_ts and self.current_timestamp < self.end_moment:
      self._total_snapshots += self._snapshot_idx
      if self._end_cutted_snapshots:
        self._finished_snapshots = True
      else:
        logger.info(f"Reload snapshot file: total-snapshots={self._total_snapshots}")
        self._snapshots_df, self.__limit_snapshot = self._read_snapshots(self._snapshot_file, self._total_snapshots)
        self._snapshot_idx = 0
        self.current_last_snapshot_ts = self._snapshots_df.iloc[-1][0]

        self._snapshot_generator = self._load_snapshot()

  def _reload_trades_df(self):
    if self.current_timestamp >= self.current_last_trade_ts and self.current_timestamp < self.end_moment:
      self._total_trades += self._trades_idx
      if self._end_cutted_trades:
        self._finished_trades = True
      else:
        logger.info(f"Reload _trades file: total-_trades={self._total_trades}")
        self._trades_df, self._limit_trades = self._read_trades(self._trades_file, self._total_trades)
        self._trades_idx = 0
        self.current_last_trade_ts = self._trades_df.iloc[-1].timestamp

        self._trade_generator = self.__load_trade()

  def total(self):
    total = len(self._snapshots_df)
    if self._trades_file is not None:
      total += len(self._trades_df)
    return total - 2

  def read_initial_moment(self, snapshot_file:str) -> datetime.datetime:
    df = pd.read_csv(snapshot_file, header=None, nrows=1)
    df = fix_timestamp_drop_millis(df, 0, 1)
    return df.loc[0, 0]

  def _read_snapshots(self, snapshot_file: str, skiprows:int) -> Tuple[pd.DataFrame, int]:
    df, length = super()._read_snapshots(snapshot_file, skiprows)
    df.index = pd.DatetimeIndex(df[0])
    df2 = df[(df.index >= self.initial_moment) & (df.index <=self.end_moment)]

    if len(df2) == 0:
      self._total_snapshots += length
      return self._read_snapshots(snapshot_file, skiprows + length)
    else:
      self._end_cutted_snapshots = df2.index[-1] < df.index[-1]
      idx = df.index.get_loc(df2.index[0])  # adjust future skiprows
      idx = idx.start if isinstance(idx, slice) else idx
      self._total_snapshots += idx
      del df

    return df2, len(df2)

  def _read_trades(self, trades_file: str, skiprows: int) -> Tuple[pd.DataFrame, int]:
    df, length = super()._read_trades(trades_file, skiprows)
    df.index = pd.DatetimeIndex(df.timestamp)
    df2 = df[(df.index >= self.initial_moment) & (df.index <=self.end_moment)]
    if len(df2) == 0:
      self._total_trades += length
      return self._read_trades(trades_file, skiprows + length)
    else:
      self._end_cutted_trades = df2.index[-1] < df.index[-1]
      idx = df.index.get_loc(df2.index[0]) # adjust future skiprows
      idx = idx.start if isinstance(idx, slice) else idx
      self._total_trades += idx
      del df

    return df2, len(df2)
