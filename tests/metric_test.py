import time
import unittest
from typing import List

import numpy as np

import test_utils
from hft.backtesting import backtest
from hft.backtesting.backtest import Backtest
from hft.backtesting.output import StorageOutput
from hft.backtesting.readers import OrderbookReader, TimeLimitedReader
from hft.backtesting.strategy import CalmStrategy
from hft.units.metric import ZNormalized
from hft.units.metrics.composite import Lipton
from hft.units.metrics.instant import VWAP_volume, HayashiYoshido, LiquiditySpectrum, CraftyCorrelation
from hft.units.metrics.time import DeltaTimeMetric
from hft.utils.consts import QuoteSides
from hft.utils.data import OrderBook


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

    snapshot = test_utils.get_orderbooks(1, src='resources/orderbook/_orderbooks.csv.gz')[0]
    values = vwap.evaluate(snapshot)
    latest = vwap.latest
    self.assertEqual(values, tuple([latest[v] for v in volumes]))

  def test_delta_lipton_metric(self):
    reader = OrderbookReader(snapshot_file='resources/orderbook/_orderbooks.csv.gz', stop_after=5000, depth_to_load=8)

    delta10 = DeltaTimeMetric(seconds=60)
    lipton = Lipton('__delta-60')
    simulation = CalmStrategy(time_metrics_snapshot=[delta10], composite_metrics=[lipton])
    metric_map = simulation.metrics_map
    lipton.set_metric_map(metric_map)
    self.assertEqual(metric_map['__delta-60'], delta10)

    first = reader._snapshot
    backtester = backtest.Backtest(reader, simulation)
    backtester.run()
    last = reader._snapshot

    # self.assertTrue((last.timestamp - first.timestamp).seconds > 60)
    # TODO: REFACTOR DELTA TO QUANTITY LIMITED METRIC (NOT TIME LIMITED)
    storage = metric_map['__delta-60'].storage
    ask_pos_xbtusd = storage[('XBTUSD', QuoteSides.ASK, 'pos')]
    ask_neg_xbtusd = storage[('XBTUSD', QuoteSides.ASK, 'neg')]
    bid_pos_xbtusd = storage[('XBTUSD', QuoteSides.BID, 'pos')]

    latest = metric_map['__delta-60'].latest
    quantity_ask_pos = latest['quantity', 'XBTUSD',   QuoteSides.ASK, 'pos']
    quantity_ask_neg = latest['quantity', 'XBTUSD',   QuoteSides.ASK, 'neg']
    volume_ask_pos = latest['volume_total', 'XBTUSD', QuoteSides.ASK, 'pos']
    volume_ask_neg = latest['volume_total', 'XBTUSD', QuoteSides.ASK, 'neg']

    self.assertEqual(volume_ask_neg, np.sum(ask_neg_xbtusd))
    self.assertEqual(volume_ask_pos, np.sum(ask_pos_xbtusd))
    self.assertEqual(quantity_ask_pos, len(ask_pos_xbtusd))
    self.assertEqual(quantity_ask_neg, len(ask_neg_xbtusd))

    replenishment = storage[(last.symbol, QuoteSides.ASK, 'pos')]
    depletion = storage[(last.symbol, QuoteSides.BID, 'neg')]
    length = min(len(depletion), len(replenishment))
    lipton_latest = metric_map['lipton'].latest[last.symbol]
    p_xy = np.corrcoef(list(depletion)[-length:], list(replenishment)[-length:])[0, 1]

    x = float(last.bid_volumes[0])
    y = float(last.ask_volumes[0])
    sqrt_corr = np.sqrt((1 + p_xy) / (1 - p_xy))
    p = 0.5 * (1. - np.arctan(sqrt_corr * (y - x) / (y + x)) / np.arctan(sqrt_corr))
    self.assertAlmostEqual(p, lipton_latest, delta=1e-2) # todo: fix it later
    print(p)

  def test_hoyashi_yoshido(self):
    hy_values = []
    lipton_values = []
    def store_only_yoshido(labels, ts, object):
      if 'hoyashi-yoshido' in labels:
        hy_values.append((ts, object))
      elif 'lipton' in labels:
        lipton_values.append((ts, object))

    reader = TimeLimitedReader(snapshot_file='resources/orderbook/_orderbooks.csv.gz', limit_time='5 min')
    hy = HayashiYoshido(seconds=20)
    lipton= Lipton(hy.name)

    simulation = CalmStrategy(delta_metrics=[hy], composite_metrics=[lipton])
    storage = StorageOutput([hy.name], [])
    storage.consume = store_only_yoshido
    t1 = time.time()
    backtester = backtest.Backtest(reader, simulation, storage)
    backtester.run()
    t2 = time.time() - t1
    print(f'ok, {t2} seconds')

  def test_liquidity_spectrum(self):
    liquidity_spectrum = LiquiditySpectrum()
    reader = OrderbookReader(snapshot_file='resources/orderbook/_orderbooks.csv.gz', nrows=10, stop_after=2)
    ob: OrderBook = next(reader)[0]
    lss = liquidity_spectrum.evaluate(ob)

    self.assertEqual(lss[0,0], np.sum(ob.ask_volumes[:3]))
    self.assertEqual(lss[1,0], np.sum(ob.ask_volumes[3:6]))
    self.assertEqual(lss[2,0], np.sum(ob.ask_volumes[6:]))

    self.assertEqual(lss[0,1], np.sum(ob.bid_volumes[:3]))
    self.assertEqual(lss[1,1], np.sum(ob.bid_volumes[3:6]))
    self.assertEqual(lss[2,1], np.sum(ob.bid_volumes[6:]))

  def test_z_normalization_simple(self):

    period = 50
    znorm = ZNormalized(period, lambda: None)
    ints = np.random.randint(10, 50, 300)

    for t in ints:
      znorm['XBTUSD'] = np.array([t])

    mu, sigma = ints[-period:].mean(), ints[-period:].std()

    v = ints[-1]
    expected = (v - mu) / sigma
    result = znorm['XBTUSD']
    self.assertAlmostEqual(expected, result, msg='Primitives must be equal')

  def test_z_normalization_tensor(self):
    period = 50
    znorm = ZNormalized(period, lambda: None)
    ints = np.random.randint(10, 50, 6000)
    ints = ints.reshape((100, 6, 10))

    for t in ints:
      znorm['XBTUSD'] = t

    mu, sigma = ints[-period:].mean(axis=0), ints[-period:].std(axis=0)

    v = ints[-1]
    expected = (v - mu) / sigma
    result = znorm['XBTUSD']
    self.assertTrue((expected == result).all(), msg='np arrays must be equal')

  def test_z_normalization_metric(self):
    period = 40

    reader = OrderbookReader(snapshot_file='resources/orderbook/orderbooks.csv.gz', nrows=10000, stop_after=1000)
    vwap_normalized = VWAP_volume([int(2.5e5), int(1e6)], name='vwap_normalized', z_normalize=period)
    vwap = VWAP_volume([int(2.5e5), int(1e6)], name='vwap')
    strategy = CalmStrategy(instant_metrics=[vwap, vwap_normalized])
    backtest = Backtest(reader, strategy)

    obs = []
    vwap_values = []
    for idx, (item, flag) in enumerate(reader):
      backtest._process_event(item, flag)
      if flag and item.symbol == 'XBTUSD':
        assert (abs(vwap.evaluate(item) - vwap_normalized.latest._storage['XBTUSD'][-1]) < 1e-4).all()
        vwap_values.append(vwap.latest[item.symbol])
        obs.append(item)


    normalized = vwap_normalized.latest['XBTUSD']
    vwap_values = vwap_values[-period:]
    obs = obs[-period:]
    vwap_values = np.array(vwap_values, dtype=np.float)
    # vwap_values = vwap_values.reshape((len(vwap_values), -1))
    mu, sigma = np.mean(vwap_values, axis=0), np.std(vwap_values, axis=0)

    v = vwap.latest['XBTUSD']
    # sh = v.shape
    # v2 = v.reshape(-1)
    v = (v - mu) / (sigma + 1e-4)
    # v2 = v2.reshape(sh)

    self.assertTrue((v == normalized).all()) # todo: does not work, wtf ??? Due to floating error, random 7.6 e-5 mistakes

  def test_crafty_correlation(self):
    crafty = CraftyCorrelation(40, 5, 'crafty')
    reader = OrderbookReader(snapshot_file='resources/orderbook/orderbooks.csv.gz', nrows=10000, stop_after=3000)
    strategy = CalmStrategy(delta_metrics=[crafty])
    backtest = Backtest(reader, strategy)
    backtest.run()

    self.assertTrue(all(map(lambda x: abs(x) <= 1.0, crafty.latest.values())))



if __name__ == '__main__':
  unittest.main()
