from dataloader.utils.data import Snapshot
import pandas as pd
import numpy as np

class Reader:

  class Row:
    pass

  def __iter__(self):
    return self

  def __next__(self):
    pass

class SnapshotReader(Reader): # todo: test

  def __init__(self, file, nrows=10000):
    """
    :param nrows: 10000 - number of rows to read per update
    :param file:
    """
    self.idx = 0
    self.target = file
    self.nrows = nrows
    self.df: pd.DataFrame = self.__read_csv()
    self.limit = len(self.df)

  def __read_csv(self):
    return pd.read_csv(self.file, compression='gzip', sep=',',
                       quotechar='"', error_bad_lines=False,
                       skiprows=self.idx + 1, nrows=self.nrows)

  def __sort_side(self, side: np.array, reverse=False): # todo: test it
    # prices: np.array = side[Snapshot.price_indices]
    price_idxs = list(Snapshot.price_indices)
    sorted(price_idxs, key = lambda idx: side[idx], reverse=reverse)
    return


  def __next__(self) -> Snapshot:

    if self.idx + 1 >= self.nrows:
      self.df = self.__read_csv()
      self.idx = 0
      self.limit = len(self.df)

    row = self.df.iloc[self.idx, :]
    timestamp = row[0] # todo: check is it correct type
    market = row[1]
    bids = row[2:52].values
    asks = row[52:].values

    bids = self.__sort_side(bids, False)
    asks = self.__sort_side(asks, True)

    self.idx += 1
    if self.limit != self.nrows and self.idx == self.limit:
      raise StopIteration

    return Snapshot(market, timestamp, bids, asks)
