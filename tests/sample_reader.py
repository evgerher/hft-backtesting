from typing import List
from utils import helper
from utils.data import OrderBook
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