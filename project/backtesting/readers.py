from utils import helper
from utils.data import Snapshot
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger('<reader>', 'DEBUG')

class Reader:

  class Row:
    pass

  def __iter__(self):
    return self

  def __next__(self):
    pass

class SnapshotReader(Reader):

  def __init__(self, file:str, nrows:int=10000, compression='infer', stop_after:int=None):
    """
    :param file: to read
    :param nrows: 10000 - number of rows to read per update
    :param stop_after: stop iteration over dataset after amount of iterations
    """
    self.idx = 0
    self.file = file
    self.nrows = nrows
    self.compression = compression

    self.df: pd.DataFrame = self.__read_csv()
    self.limit = len(self.df)
    self.total = 0
    self.stop_after = stop_after

  def __read_csv(self):
    return pd.read_csv(self.file, header=None, compression=self.compression, sep=',',
                       quotechar='"', error_bad_lines=False,
                       skiprows=self.idx, nrows=self.nrows)


  def __next__(self) -> Snapshot:

    if self.stop_after is not None and self.idx == self.stop_after:
      raise StopIteration

    if self.limit != self.nrows and self.idx == self.limit:
      self.total += self.idx
      logger.debug(f"Finished file {self.file}, read {self.total} rows")
      raise StopIteration

    if self.idx + 1 >= self.nrows:
      self.total += self.idx
      self.df = self.__read_csv()
      self.idx = 0
      self.limit = len(self.df)

    row: pd.Series = self.df.iloc[self.idx, :]
    timestamp, market, bids, asks = helper.snapshot_line_parser(row)

    self.idx += 1

    return Snapshot.from_sides(timestamp, market, bids, asks)

  def __str__(self):
    return f'<Snapshot reader on file={self.file}, batch_nrows={self.nrows}>'
