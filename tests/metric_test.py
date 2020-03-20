import unittest
from typing import List

import numpy as np
import sample_reader
from backtesting import backtest
from backtesting.readers import SnapshotReader, OrderbookReader
from backtesting.strategy import CalmStrategy
from metrics.metrics import VWAP_volume, DeltaMetric, Lipton
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

  def test_vwap_latest(self):

    volumes = list(map(lambda x: int(x), [5e5, 1e6, 1e6+5e5]))
    vwap = VWAP_volume(volumes)
    self.assertListEqual(volumes, vwap.subitems())

    snapshot = sample_reader.get_orderbooks(1, src='resources/orderbook10/orderbook.csv')[0]
    values = vwap.evaluate(snapshot)
    latest = vwap.latest
    self.assertEqual(values, tuple([latest[v] for v in volumes]))

  def test_delta_lipton_metric(self):
    reader = OrderbookReader(snapshot_file='resources/orderbook_fixed/orderbooks.csv.gz', stop_after=5000, depth_to_load=5)

    delta10 = DeltaMetric(seconds=10)
    lipton = Lipton('delta-10')
    simulation = CalmStrategy(time_metrics_snapshot=[delta10], composite_metrics=[lipton])
    metric_map = simulation.metrics_map
    lipton.set_metric_map(metric_map)
    self.assertEqual(metric_map['delta-10'], delta10)

    first = reader._snapshot
    backtester = backtest.Backtest(reader, simulation)
    backtester.run()
    last = reader._snapshot

    self.assertTrue((last.timestamp - first.timestamp).seconds > 10)

    storage = metric_map['delta-10'].storage
    ask_pos_xbtusd = storage[('XBTUSD', 'ask', 'pos')]
    ask_neg_xbtusd = storage[('XBTUSD', 'ask', 'neg')]
    bid_pos_xbtusd = storage[('XBTUSD', 'bid', 'pos')]
    bid_neg_xbtusd = storage[('XBTUSD', 'bid', 'neg')]

    latest = metric_map['delta-10'].latest

    quantity_ask_pos = latest['quantity', 'XBTUSD', 'ask', 'pos']
    quantity_ask_neg = latest['quantity', 'XBTUSD', 'ask', 'neg']
    volume_ask_pos = latest['volume', 'XBTUSD', 'ask', 'pos']
    volume_ask_neg = latest['volume', 'XBTUSD', 'ask', 'neg']

    self.assertEqual(volume_ask_neg, np.sum(ask_neg_xbtusd))
    self.assertEqual(volume_ask_pos, np.sum(ask_pos_xbtusd))
    self.assertEqual(quantity_ask_pos, len(ask_pos_xbtusd))
    self.assertEqual(quantity_ask_neg, len(ask_neg_xbtusd))

if __name__ == '__main__':
  unittest.main()