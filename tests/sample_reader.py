from typing import List
import numpy as np

from utils import helper
from utils.data import Snapshot


def read_snapshot_rows(src: str = 'resources/snapshots.csv') -> List[str]:
  with open(src, 'r') as f:
    return f.read().split('\n')

def get_snapshots(limit: int = None, src: str = 'resources/snapshots.csv') -> List[Snapshot]:
  def line_to_snapshot(line: str) -> Snapshot:
    items = line.split(',')
    date, symbol, bids, asks = helper.snapshot_line_parser(items)
    return Snapshot.from_sides(date, symbol, bids, asks)

  rows = read_snapshot_rows(src)
  if limit is None:
    limit = len(rows)
  return list(map(line_to_snapshot, rows[:limit]))
