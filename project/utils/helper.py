import datetime
from typing import List
import numpy as np

def snapshot_line_parser(line: List):
  assert len(line) == 103
  millis = int(line[1])
  if type(line[0]) is str:
    date = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
  elif type(line[1]) is datetime:
    date = line[1]
  date = date + datetime.timedelta(milliseconds=millis)
  symbol = line[2]
  asks = np.array(line[3:53], dtype=np.float)
  bids = np.array(line[53:], dtype=np.float)

  return date, symbol, bids, asks