import unittest
import traces
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from hft.backtesting.readers import OrderbookReader
from hft.units.filters import Filters
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


  def test_deltas(self):
    filter = Filters.DepthFilter(3)
    reader = OrderbookReader('resources/orderbook/orderbooks.csv.gz', stop_after=3000)

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

    # tf, symbol, side? delta-values
    bid_ts, bid_deltas = zip(*bids)
    ask_ts, ask_deltas = zip(*asks)
    bid_deltas = np.array(bid_deltas)
    ask_deltas = np.array(ask_deltas)

    bid_deltas[1:] -= bid_deltas[:-1]
    ask_deltas[1:] -= ask_deltas[:-1]

    # bids = zip(bid_ts, np.log(bid_deltas))
    # asks = zip(ask_ts, np.log(ask_deltas))

    bids = zip(bid_ts[1:], np.clip(bid_deltas[1:], -1000., 1000.))
    asks = zip(ask_ts[1:], np.clip(ask_deltas[1:], -1000., 1000.))
    # bids = zip(bid_ts[1:], bid_deltas[1:])
    # asks = zip(ask_ts[1:], ask_deltas[1:])
    print(len(bid_ts), len(ask_ts))

    t1 =time.time()
    ts_bid = traces.TimeSeries(bids, default=0)
    ts_ask = traces.TimeSeries(asks, default=0)
    t2 = time.time() - t1
    print(t2)

    ts = ts_ask * ts_bid

    # plt.ylim(-100000, 100000)
    fig, ax = ts.plot()
    ax.set_ylim([-3000, 3000])
    ax.figure.set_size_inches(12, 10, forward=True)

    plt.show()
