import unittest
import numpy as np
import test_utils

from metrics.filters import Filters
from utils.data import OrderBook


class FilterTest(unittest.TestCase):
  def test_filter_update(self):
    _filter = Filters.DepthFilter(4)
    snapshots = test_utils.get_snapshots(30, 'resources/snapshots_filter.csv')

    filtered = list(filter(lambda x: x is not None, map(lambda x: (_filter.process(x), x), snapshots)))

    self.assertTrue(len(filtered) >= 4)

  def test_filter_depth(self):
    filter3 = Filters.DepthFilter(3)
    filter4 = Filters.DepthFilter(4)
    snapshots = test_utils.get_snapshots(10, 'resources/snapshots_depth5.csv', length=20)
    filtered3 = list(filter(lambda x: x[0] is not None, map(lambda x: (filter3.process(x), x), snapshots)))
    filtered4 = list(filter(lambda x: x[0] is not None, map(lambda x: (filter4.process(x), x), snapshots)))

    self.assertEqual(len(filtered3), 2)
    self.assertEqual(len(filtered4), 3) # one XBTUSD altered on ask side level=3

  def test_delta_bid_neg(self):
    filter3 = Filters.DepthFilter(4)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 200]))
    o2 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 100, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 200]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'bid')
    self.assertEqual(delta[0, 0], 9.0)
    self.assertEqual(delta[1, 0], -100)

  def test_delta_bid_pos(self):
    filter3 = Filters.DepthFilter(4)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 200]))
    o2 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([150, 300, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 200]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]
    self.assertEqual(result[2], 'bid')
    self.assertEqual(delta[0, 0], 10.0)
    self.assertEqual(delta[1, 0], 50)

    self.assertEqual(delta[0, 1], 9.0)
    self.assertEqual(delta[1, 1], 100)


  def test_delta_ask_neg(self):
    filter3 = Filters.DepthFilter(4)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 200]))
    o2 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 90, 200]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'ask')
    self.assertEqual(delta[0, 0], 12.0)
    self.assertEqual(delta[1, 0], -110)

  def test_delta_ask_pos(self):
    filter3 = Filters.DepthFilter(4)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))
    o2 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 1000]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'ask')
    self.assertEqual(delta[0, 0], 13.0)
    self.assertEqual(delta[1, 0], 850)

  def test_delta_ask_level_new(self):
    filter3 = Filters.DepthFilter(4)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))
    o2 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([10.0, 11.0, 12.0]), np.array([30, 200, 200]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'ask-alter')
    self.assertEqual(delta[0, 0], 10.0)
    self.assertEqual(delta[1, 0], 30)

  def test_delta_bid_level_new(self):
    filter3 = Filters.DepthFilter(4)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))
    o2 = OrderBook('test', None, np.array([11.0, 10.0, 9.0]), np.array([60, 100, 200]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'bid-alter')
    self.assertEqual(delta[0, 0], 11.0)
    self.assertEqual(delta[1, 0], 60)

  def test_delta_bid_level_consumed(self):
    filter3 = Filters.DepthFilter(3)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))
    o2 = OrderBook('test', None, np.array([9.0, 8.0, 7.0]), np.array([50, 300, 1000]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'bid-alter')
    self.assertEqual(delta[0, 0], 10.0)
    self.assertEqual(delta[1, 0], -100)

    self.assertEqual(delta[0, 1], 9.0)
    self.assertEqual(delta[1, 1], -150)

  def test_delta_ask_level_consumed(self):
    filter3 = Filters.DepthFilter(3)
    o1 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([11.0, 12.0, 13.0]), np.array([200, 200, 150]))
    o2 = OrderBook('test', None, np.array([10.0, 9.0, 8.0]), np.array([100, 200, 300]),
                   np.array([12.0, 13.0, 14.0]), np.array([130, 150, 1000]))

    filter3.process(o1)
    result = filter3.process(o2)
    delta = result[-1]

    self.assertEqual(result[2], 'ask-alter')
    self.assertEqual(delta[0, 0], 11.0)
    self.assertEqual(delta[1, 0], -200)

    self.assertEqual(delta[0, 1], 12.0)
    self.assertEqual(delta[1, 1], -70)

if __name__ == '__main__':
  unittest.main()
