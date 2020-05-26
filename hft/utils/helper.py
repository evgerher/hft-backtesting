import datetime
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from hft.utils.consts import TradeSides
from hft.utils.data import OrderBook, Trade


def snapshot_line_parser(line: Union[List], length:int=100):
  assert len(line) == length + 3
  millis = int(line[1])
  date = convert_to_datetime(line[0])
  date = date + datetime.timedelta(milliseconds=millis)
  symbol = line[2]

  assert length % 2 == 0 and length > 0
  asks = np.array(line[3:3 + length // 2], dtype=np.float)
  bids = np.array(line[3 + length // 2:], dtype=np.float)

  return date, symbol, bids, asks


def orderbook_line_parse(line: pd.Series, depth:int=10) -> OrderBook:
  target = np.array(line[2:])
  ap = target[:depth]
  av = target[10:10+depth]
  bp = target[20:20+depth]
  bv = target[30:30+depth]
  return OrderBook(line[1], line[0], bp, bv, ap, av)

def trade_line_parser(line: pd.Series) -> Trade:
  # input: pandas.Series
  return Trade(line['symbol'], line['timestamp'], line['side'], line['price'], line['volume'])


def convert_to_datetime(moment: Union[datetime.datetime, str]):
  if type(moment) is str:
    return datetime.datetime.strptime(moment, '%Y-%m-%d %H:%M:%S')
  elif type(moment) is datetime.datetime:
    return moment


def fix_timestamp(df, timestamp_index, millis_index, precomputed=False) -> pd.DataFrame:
  df.iloc[:, timestamp_index] = pd.to_datetime(df.iloc[:, timestamp_index])
  if not precomputed:
    df.iloc[:, millis_index] = df.iloc[:, millis_index].apply(lambda x: datetime.timedelta(milliseconds=x))
    df.iloc[:, timestamp_index] += df.iloc[:, millis_index]
  return df

def fix_timestamp_drop_millis(df, timestamp_index, millis_index, precomputed=False):
  df = fix_timestamp(df, timestamp_index, millis_index, precomputed)
  df = df.drop(columns=[millis_index])
  return df


def fix_trades_rename(df, timestamp_index, millis_index, precomputed=False):
  df = fix_timestamp_drop_millis(df, timestamp_index, millis_index, precomputed)
  df = df.drop(columns=[5])  # remove `action`
  df.columns = ['symbol', 'timestamp', 'price', 'volume', 'side']
  df.loc[df.side == 'Sell', "side"] = TradeSides.SELL
  df.loc[df.side == 'Buy', "side"] = TradeSides.BUY
  return df


def get_datetime_index(df: pd.DataFrame, is_orderboook: bool) -> pd.DatetimeIndex:
  if is_orderboook:
    timestamp_index = 0
    df = fix_timestamp(df, timestamp_index, 1)
  else:
    timestamp_index = 1
    df = fix_timestamp(df, timestamp_index, 2)
  return pd.DatetimeIndex(df.iloc[:, timestamp_index])


def convert_to_timedelta(time_symbol: str) -> Optional[datetime.timedelta]:
  q, symbol = time_symbol.split()
  q = int(q)
  assert q > 0

  if symbol == 'sec':
    return datetime.timedelta(seconds=q)
  elif symbol == 'msec':
    return datetime.timedelta(milliseconds=q)
  elif symbol == 'min':
    return datetime.timedelta(minutes=q)
  elif symbol == 'h':
    return datetime.timedelta(hours=q)

  return None
