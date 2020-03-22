from typing import List, Union, Dict

from backtesting.data import OrderRequest
from backtesting.readers import ListReader
from backtesting.strategy import Strategy
from metrics.metrics import InstantMetric
from utils import helper
from utils.data import OrderBook, Trade
import pandas as pd

def read_snapshot_rows(src: str = 'resources/snapshots.csv') -> List[str]:
  with open(src, 'r') as f:
    content = f.read().split('\n')
    if len(content[-1]) == 0:
      return content[:-1]
    return content

def get_snapshots(limit: int = None, src: str = 'resources/snapshots.csv', length=100) -> List[OrderBook]:
  def line_to_snapshot(line: str) -> OrderBook:
    items = line.split(',')
    date, symbol, bids, asks = helper.snapshot_line_parser(items, length=length)
    return OrderBook.from_sides(date, symbol, bids, asks)

  rows = read_snapshot_rows(src)
  if limit is None:
    limit = len(rows)
  return list(map(line_to_snapshot, rows[:limit]))

def get_orderbooks(limit: int = None, src='resouces/orderbook10/orderbook.csv') -> List[OrderBook]:
  df = pd.read_csv(src, nrows=limit, header=None)
  items = []
  for idx in range(limit):
    items.append(helper.orderbook_line_parse(df.iloc[idx, :]))
  return items

class SimpleStrategy(Strategy):
  def __init__(self, instant_metrics: List[InstantMetric], time_metrics_trade, reader: ListReader):
    super().__init__(instant_metrics, time_metrics_trade=time_metrics_trade)
    self.idx = 0
    self.reader = reader

  def define_orders(self, row: Union[Trade, OrderBook], statuses, memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
    item = []
    if self.idx == 0:
      item = [OrderRequest.create_bid(9.5, 450, 'test', self.reader[0].timestamp)]
    elif self.idx == 4:
      item = [OrderRequest.create_bid(9.5, 200, 'test', self.reader[3].timestamp)]

    self.idx += 1
    return item

