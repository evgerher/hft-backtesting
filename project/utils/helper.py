import datetime
import pandas
from typing import List, Union
import numpy as np

from utils.data import Trade


def snapshot_line_parser(line: Union[List, pandas.Series], length:int=100):
  assert len(line) == length + 3
  millis = int(line[1])
  date = convert_to_datetime(line[0])
  date = date + datetime.timedelta(milliseconds=millis)
  symbol = line[2]

  assert length % 2 == 0 and length > 0
  asks = np.array(line[3:3 + length // 2], dtype=np.float)
  bids = np.array(line[3 + length // 2:], dtype=np.float)

  return date, symbol, bids, asks


def trade_line_parser(line: Union[List, pandas.Series]) -> Trade:
  symbol = line[0]
  moment = convert_to_datetime(line[1])
  millis = int(line[2])
  moment = moment + datetime.timedelta(milliseconds=millis)
  price = line[3]
  volume = line[4]
  side = line[6]

  return Trade(symbol, moment, side, price, volume)


def convert_to_datetime(moment: Union[datetime.datetime, str]):
  if type(moment) is str:
    return datetime.datetime.strptime(moment, '%Y-%m-%d %H:%M:%S')
  elif type(moment) is datetime.datetime:
    return moment
