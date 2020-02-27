import datetime
from typing import List
import numpy as np

def snapshot_line_parser(line: List, length=100):
  # assert len(line) == 103
  millis = int(line[1])
  if type(line[0]) is str:
    date = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
  elif type(line[1]) is datetime:
    date = line[1]
  date = date + datetime.timedelta(milliseconds=millis)
  symbol = line[2]

  assert length % 2 == 0 and length > 0
  asks = np.array(line[3:3 + length // 2], dtype=np.float)
  bids = np.array(line[3 + length // 2:], dtype=np.float)

  return date, symbol, bids, asks