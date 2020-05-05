from abc import ABC

import concurrent
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import Future

from typing import Tuple

import pandas as pd
import numpy as np
import os
import datetime
from hft.utils.logger import  setup_logger
from hft.utils import helper

logger = setup_logger('<sampler>')


class Sampler(ABC):
  # todo: separate samplers and writers logic
  # todo: implement sampler as iterator

  def __init__(self, orderbooks_file: str, trades_file: str, destination: str, max_workers=4, nrows=1000000, starting_idx=0):
    self._orderbooks_file = orderbooks_file
    self._trades_file = trades_file
    self._destination = destination
    self.__pool = ThreadPoolExecutor(max_workers=max_workers)
    self.__file_idx = starting_idx
    self._nrows = nrows
    self._orderbook_skiprows = 0
    self._trade_skiprows = 0


    orderbook_f: Future = self.__pool.submit(self._load_frame, self._orderbooks_file, nrows, self._orderbook_skiprows, True)
    trades_f: Future = self.__pool.submit(self._load_frame, self._trades_file, nrows, self._trade_skiprows, False)

    self._orderbooks: pd.DataFrame = orderbook_f.result()
    self._trades: pd.DataFrame = trades_f.result()
    # self._orderbooks = self._load_frame(self._orderbooks_file, nrows, self._orderbook_skiprows, True)
    # self._trades = self._load_frame(self._trades_file, nrows, self._trade_skiprows, False)

    self._trade_skiprows += len(self._trades)
    self._orderbook_skiprows += len(self._orderbooks)

    if not os.path.exists(destination):
      os.mkdir(destination)

    self._futures = []

    self._finished_orderbook = False
    self._finished_trade = False

  def _sample(self) -> Tuple[Tuple[np.array, bool], Tuple[np.array, bool]]:
    '''

    :return: [tuple of indexes and is_done flag for dataframe] per _orderbooks and _trades
    '''
    raise NotImplementedError()

  def _load_frame(self, fname: str, nrows: int, skiprows: int, is_orderbook:  bool) -> pd.DataFrame:
    df = pd.read_csv(fname, nrows=nrows, skiprows=skiprows, header=None)
    df.index = helper.get_datetime_index(df, is_orderbook)
    if not is_orderbook:
      df.columns = ['symbol', 'timestamp', 'millis', 'price', 'volume', 'action', 'side']
    return df

  def _split_by_sample(self, orderbook_samples, trade_samples):

    def write_csv(frame: pd.DataFrame, fname: str):
      frame.to_csv(f'{self._destination}/{fname}.csv.gz', index=False, header=False, compression='gzip')

    def callback(future):
      logger.info(f'Task for file {future.custom_name} finished')

    orderbook_samples: pd.DataFrame = self._orderbooks[orderbook_samples]
    trade_samples: pd.DataFrame = self._trades[trade_samples]

    for frame, fname in zip([orderbook_samples, trade_samples], [f'orderbook_{self.__file_idx}', f'trade_{self.__file_idx}']):
      f = self.__pool.submit(write_csv, frame, fname)
      f.custom_name = fname
      f.add_done_callback(callback)
      self._futures.append(f)

    self.__file_idx += 1

  def _reload_orderbook(self):
    orderbooks = self._load_frame(self._orderbooks_file, self._nrows, self._orderbook_skiprows, True)
    self._orderbook_skiprows += len(orderbooks)

    # wait until _futures empty
    concurrent.futures.wait(self._futures)
    self._futures.clear()
    logger.info("Reload orderbook df")
    self._orderbooks = orderbooks
    self.current_moment = self._orderbooks.index[0]

  def _reload_trade(self):
    trades = self._load_frame(self._trades_file, self._nrows, self._trade_skiprows, False)
    self._trade_skiprows += len(trades)

    # wait until _futures empty
    concurrent.futures.wait(self._futures)
    self._futures.clear()
    logger.info("Reload trade df")
    self._trades = trades


  def split_samples(self):
    try:
      while not self._finished_trade and not self._finished_orderbook:
        (orderbook_samples, orderbooks_done), (trade_samples, trades_done) = self._sample()

        if len(self._trades) != self._nrows and trades_done:
          self._finished_trade = True

        if len(self._orderbooks) != self._nrows and orderbooks_done:
          self._finished_orderbook = True

        self._split_by_sample(orderbook_samples, trade_samples)

        if orderbooks_done and not self._finished_orderbook: # flag whether dataframe is finished and requires reload
          self._reload_orderbook()
        if trades_done and not self._finished_trade:
          self._reload_trade()
      concurrent.futures.wait(self._futures)
    finally:
      self.__pool.shutdown()
      logger.info('Finished processing')

class TimeSampler(Sampler):
  def __init__(self, orderbooks_file: str, trades_file: str, destination: str, seconds: int, **kwargs):
    # todo: warmup ?
    super().__init__(orderbooks_file, trades_file, destination, **kwargs)
    self.__delta = datetime.timedelta(seconds=seconds)
    self.current_moment = self._orderbooks.index[0]
    logger.info("Loaded TimeSampler")

  def _sample(self) -> Tuple[Tuple[np.array, bool], Tuple[np.array, bool]]:
    orderbook_samples = (self._orderbooks.index >= self.current_moment) & (self._orderbooks.index < self.current_moment + self.__delta)
    trade_samples = (self._trades.index >= self.current_moment) & (self._trades.index < self.current_moment + self.__delta)
    self.current_moment += self.__delta
    ob_done = self._orderbooks.index[-1] <= self.current_moment
    tr_done = self._trades.index[-1] <= self.current_moment

    return (orderbook_samples, ob_done), (trade_samples, tr_done)


class VolumeSampler(Sampler):
  def __init__(self, orderbooks_file: str, trades_file: str, destination: str, volume: int, target_symbol: str, **kwargs):
    super().__init__(orderbooks_file, trades_file, destination, **kwargs)
    self.volume = volume
    self.target_symbol = target_symbol

    self._traded_volume = 0.0
    self._traded_cusum = self._trades[self._trades.symbol == self.target_symbol].volume.cumsum()
    self._from = self._traded_cusum.index[0]
    self.current_moment = self._trades.index[0]
    logger.info("Loaded VolumeSampler")

  def _reload_trade(self):
    super()._reload_trade()
    self._traded_cusum = self._trades[self._trades.symbol == self.target_symbol].volume.cumsum()
    self._traded_volume = 0.0
    self._from = self._traded_cusum.index[0]

  def _sample(self) -> Tuple[Tuple[np.array, bool], Tuple[np.array, bool]]:

    idx = self._traded_cusum >= self.volume + self._traded_volume

    if idx.any():
      idx = np.argmax(idx)
    else:
      idx = self._traded_cusum.index[-1]

    # idx = np.argmax() or len(self._traded_cusum) - 1 # prevent None
    volume = self._traded_cusum[idx]
    if type(volume) == pd.Series: # pd.Series
      volume = volume[-1]
    self._traded_volume += volume

    # Get first next
    # t = self._trades.index[idx]
    until = self._trades.index[self._trades.index > idx]
    if len(until) > 0:
      until = until[0]
    else:
      until = idx

    # Define horizon
    ob_sample = (self._orderbooks.index >= self._from) & (self._orderbooks.index < until)
    tr_sample = (self._trades.index >= self._from) & (self._trades.index < until)

    self._from = until
    ob_done = until >= self._orderbooks.index[-1]
    tr_done = until >= self._traded_cusum.index[-1]

    return (ob_sample, ob_done), (tr_sample, tr_done)
