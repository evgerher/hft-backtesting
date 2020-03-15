import unittest
from typing import List

import numpy as np
import sample_reader
from metrics.metrics import VWAP_volume
from utils.data import OrderBook


class MetricTest(unittest.TestCase):
  def test_vwap_volume(self):
    vwap = VWAP_volume([5, 20, 30])

    ap = np.array([5, 6, 7], dtype=np.float)
    av = np.array([20, 10, 10], dtype=np.int)

    bp = np.array([4, 3, 2], dtype=np.float)
    bv = np.array([15, 10, 12], dtype=np.int)

    snapshot = OrderBook('test', None, bp, bv, ap, av)
    values_ask = vwap._evaluate_side(snapshot.ask_prices, snapshot.ask_volumes)
    values_bid = vwap._evaluate_side(snapshot.bid_prices, snapshot.bid_volumes)

    self.assertListEqual([5.0, 5.0, 16./3], values_ask.tolist())
    self.assertListEqual([4, 3.75, 10./3], values_bid.tolist())

    values: List[np.array] = vwap.evaluate(snapshot)
    self.assertListEqual(values[0].tolist(), values_bid.tolist())
    self.assertListEqual(values[1].tolist(), values_ask.tolist())

  def test_vwap_increasing(self):
    def almost_equal(value_1, value_2, accuracy=10 ** -3):
      return abs(value_1 - value_2) < accuracy

    ap = np.array([5, 6, 7], dtype=np.float)
    av = np.array([10, 20, 30], dtype=np.int)

    bp = np.array([4, 3, 2], dtype=np.float)
    bv = np.array([5, 10, 15], dtype=np.int)

    vwap = VWAP_volume([5, 20, 30])
    snapshot = OrderBook('test', None, bp, bv, ap, av)
    values: List[np.array] = vwap.evaluate(snapshot)

    bid_values = values[0]
    ask_values = values[1]
    assert all(almost_equal(*values) for values in zip(bid_values, [4.0, 3.0, 8./3]))
    assert all(almost_equal(*values) for values in zip(ask_values, [5.0, 5.5, 17./3]))

  def test_vwap_real(self):

    vwap = VWAP_volume(list(map(lambda x: int(x), [5e5, 1e6, 1e6+5e5])))
    snapshot = sample_reader.get_orderbooks(1, src='resources/orderbook10/orderbook.csv')[0]
    values = vwap.evaluate(snapshot)
    print(values)
    self.assertEqual(True, True)

if __name__ == '__main__':
  unittest.main()