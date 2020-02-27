from typing import List
from utils import helper
from utils.data import Snapshot


def read_snapshot_rows(src: str = 'resources/snapshots.csv') -> List[str]:
  with open(src, 'r') as f:
    content = f.read().split('\n')
    if len(content[-1]) == 0:
      return content[:-1]
    return content

def get_snapshots(limit: int = None, src: str = 'resources/snapshots.csv', length=100) -> List[Snapshot]:
  def line_to_snapshot(line: str) -> Snapshot:
    items = line.split(',')
    date, symbol, bids, asks = helper.snapshot_line_parser(items, length=length)
    return Snapshot.from_sides(date, symbol, bids, asks)

  rows = read_snapshot_rows(src)
  if limit is None:
    limit = len(rows)
  return list(map(line_to_snapshot, rows[:limit]))
