import pandas as pd
import numpy as np

def restore_indexes():
  df = pd.read_csv('resources/may1/index.CSV', names=['symbol', 'date', 'index'])
  df['date'] = pd.to_datetime(df.date)
  millis = df.date.astype(np.int64) / int(1e6)
  seq = millis.rolling(2).apply(lambda x: x[1] - x[0])
  idxs = seq > 60000

  dates = df.loc[idxs].date.apply(str).tolist() + [str(df.loc[len(df) -1].date)]

  idxs = np.argwhere(idxs == True).flatten()
  idxs = (idxs).tolist()
  idxs = [0] + idxs + [len(df) - 1]
  for idx, (i, j) in enumerate(zip(idxs[:-1], idxs[1:])):
    if (j != len(df) - 1):
      j -=1
    partial = df.loc[i:j]
    partial.to_csv(f'resources/may1/indexes/{_from}.csv', index=False, header=None)
    print(f'Saved: {dates[_from]} as `resources/may1/indexes/{_from}.csv`')

def read_partial(fname, skiprows, nrows):
  return pd.read_csv(fname, skiprows=skiprows, nrows=nrows, header=None)

def search_index_jump(fname, didx):
  nrows = 500000

  skipped = 0
  # 1. `2020-03-10 18:25:00 - 2020-03-24 15:30:00`
  # 2. `2020-03-27 06:19:00 - 2020-04-28 15:34:00`
  # 3. `2020-04-28 15:44:00 - 2020-04-29 12:31:00`
  # 4. `2020-04-29 12:36:00 - 2020-05-01 14:39:00`

  breaks = ['2020-03-27 06:19:00', '2020-04-28 15:44:00', '2020-04-29 12:36:00']
  breaks = pd.to_datetime(breaks).tolist()

  while True:
    df = read_partial(fname, skipped, nrows)
    dates = pd.to_datetime(df[0])

    dates_cond = dates >= breaks[didx]
    if dates_cond.any():
      first = np.argmax(dates_cond == True)
      print(f'Row index = {skipped + first}')
      didx += 1
      break
    else:
      skipped += nrows
      print(f'Skipped: {skipped}')


def main():
  # fname = 'resources/may1/_trades.CSV.gz'
  fname = 'resources/may1/_orderbooks.CSV.gz'
  # search_index_jump(fname, 0)
  df = read_partial(fname, 130357640 + 270000000, 100000)
  # from 1 until 130357640
  print('ok')

if __name__ == '__main__':
  main()
