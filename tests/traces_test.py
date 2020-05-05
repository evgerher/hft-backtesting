import unittest
import traces
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters

from hft.backtesting.readers import TimeLimitedReader
from hft.units.filters import Filters

register_matplotlib_converters()
import time

class TracesTest(unittest.TestCase):
  def test_traces(self):
    ts = traces.TimeSeries(default=0)
    ts[datetime(2042, 2, 1,  6,  0,  0)] = 10
    ts[datetime(2042, 2, 1,  7,  45,  56)] = 10
    print(ts)
    f = ts[datetime(2042, 2, 1,  6,  0,  0)]
    f2 = ts[datetime(2042, 2, 1,  6,  10,  0)]
    print(f, f2)
    # del ts[ts.first_key()]
    print(ts)

    ts2 = traces.TimeSeries(default=0)
    ts2[datetime(2042, 2, 1,  5,  0,  0)] = 5
    ts2[datetime(2042, 2, 1,  6,  45,  0)] = 5
    ts2[datetime(2042, 2, 1,  7,  40,  0)] = 2
    ts2[datetime(2042, 2, 1,  8,  0,  0)] = 10

    # h = traces.Histogram().median()
    count = ts * ts2
    # d = count.plot(figure_width=6,interpolate=)
    count.compact()
    count.plot()
    l = list(count.iterintervals(n=1))
    # count = traces.TimeSeries.merge([ts, ts2], operation=sum)
    # hist = count.distribution()
    # a, b= hist.items()
    # plt.hist(hist.items())
    # plt.show()

    self.assertEqual(count[datetime(2042, 2, 1, 7, 50)], 20)
    self.assertEqual(count[datetime(2042, 2, 1, 8)], 100)

  def test_timelimited_reader(self):
    reader = TimeLimitedReader('resources/orderbook/_orderbooks.csv.gz', limit_time='5 min', trades_file='resources/orderbook/_trades.csv.gz')
    snapshot_df = reader._snapshots_df
    trades_df = reader._trades_df
    initial_moment = reader.initial_moment

    last_trade_ts = trades_df.iloc[-1]['timestamp']
    last_snapshot_ts = snapshot_df.iloc[-1][0]

    delta = timedelta(minutes=5)


    self.assertTrue(initial_moment + delta >= last_trade_ts)
    self.assertTrue(initial_moment + delta >= last_snapshot_ts)


  def test_deltas(self):
    filter = Filters.DepthFilter(3)
    reader = TimeLimitedReader('resources/orderbook/_orderbooks.csv.gz', skip_time='530 sec', limit_time='10 sec')

    bids = []
    asks = []
    for item, _ in reader:
      if item.symbol == 'XBTUSD':
        res = filter.process(item)
        if res is not None and res[-2] != -1:
          v = np.sum(res[-1][1,:])
          if res[-2] % 2 == 0 and v < 0:
            bids.append((res[0], -v))
          elif v > 0:
            asks.append((res[0], v))

    # tf, symbol, side? __delta-values
    bid_ts, bid_deltas = zip(*bids)
    ask_ts, ask_deltas = zip(*asks)
    bid_deltas = np.array(bid_deltas)
    ask_deltas = np.array(ask_deltas)

    bid_deltas[1:] -= bid_deltas[:-1]
    ask_deltas[1:] -= ask_deltas[:-1]

    # bids = zip(bid_ts, np.log(bid_deltas))
    # asks = zip(ask_ts, np.log(ask_deltas))

    # bids = zip(bid_ts[1:], np.clip(bid_deltas[1:], -1000., 1000.))
    # asks = zip(ask_ts[1:], np.clip(ask_deltas[1:], -1000., 1000.))
    bids = zip(bid_ts[1:], bid_deltas[1:])
    asks = zip(ask_ts[1:], ask_deltas[1:])
    print(f'bid deltas (neg) {len(bid_ts)}, ask deltas (pos) = {len(ask_ts)}')

    t1 = time.time()
    ts_bid = traces.TimeSeries(bids, default=0)
    ts_ask = traces.TimeSeries(asks, default=0)
    ts = ts_ask * ts_bid
    t2 = time.time() - t1
    print(f'Time to compute timeseries {t2}')

    ts.plot()
    plt.show()

    t1 = time.time()
    sq1 = np.sqrt(np.sum(np.square(ask_deltas[1:])))
    sq2 = np.sqrt(np.sum(np.square(bid_deltas[1:])))
    values = sum(list(ts._d.values()))
    hy = values / sq1 / sq2
    t2 = time.time() - t1
    print(f'Time to compute Hoyashi-Yoshido cor {t2}')
    print(hy)

  def test_traces_am(self):
    ts = traces.TimeSeries(default=0)
    ts2 = traces.TimeSeries(default=0)
    ts.set_interval(start=datetime(year=2000, month=1, day=1, hour=4), end=datetime(year=2000, month=1, day=1, hour=5), value=5)
    ts.set_interval(start=datetime(year=2000, month=1, day=1, hour=5), end=datetime(year=2000, month=1, day=1, hour=6), value=10)
    ts.set_interval(start=datetime(year=2000, month=1, day=1, hour=6), end=datetime(year=2000, month=1, day=1, hour=7), value=3)


    ts2.set_interval(start=datetime(year=2000, month=1, day=1, hour=3, minute=30), end=datetime(year=2000, month=1, day=1, hour=4, minute=30), value=8)
    ts2.set_interval(start=datetime(year=2000, month=1, day=1, hour=4, minute=30), end=datetime(year=2000, month=1, day=1, hour=5, minute=30), value=5)
    ts2.set_interval(start=datetime(year=2000, month=1, day=1, hour=5, minute=30), end=datetime(year=2000, month=1, day=1, hour=6, minute=30), value=3)
    ts2.set_interval(start=datetime(year=2000, month=1, day=1, hour=6, minute=30), end=datetime(year=2000, month=1, day=1, hour=7, minute=30), value=5)

    # ts.plot()
    # ts2.plot()
    ts3 = ts * ts2
    # ts3.plot()
    # plt.show()

    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=3, minute=50)], 0)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=4, minute=10)], 40)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=4, minute=40)], 25)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=5, minute=10)], 50)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=5, minute=40)], 30)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=6, minute=10)], 9)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=6, minute=40)], 15)
    self.assertEqual(ts3[datetime(year=2000, month=1, day=1, hour=7, minute=10)], 0)
