import unittest

from backtesting import readers, backtest
from backtesting.output import TestOutput
from backtesting.readers import ListReader
from backtesting.strategy import Strategy
from backtesting.data import OrderStatus, OrderRequest

from metrics.filters import Filters
from metrics.metrics import *


class BacktestTest(unittest.TestCase):
  def test_run(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz', stop_after=100)
    metrics = [VWAP_depth(3),
               VWAP_volume(volumes=[int(5e5), int(1e6)], symbol='XBTUSD'),
               VWAP_volume(volumes=[int(1e5), int(5e5)], symbol='ETHUSD')]
    simulation = Strategy(metrics, [Filters.DepthFilter(3)])
    backtester = backtest.Backtest(reader, simulation, None)

    backtester.run()
    self.assertTrue(len(backtester.memory['XBTUSD']) > 2)

    # self.assertTrue(len(backtester.metrics["('XBTUSD', 'VWAP (Depth): 3 bid')"]) > 2)

  def test_init_moment(self):
    reader = readers.SnapshotReader('resources/snapshots.csv.gz')
    callables = [('trades count', lambda trades: len(trades))]
    simulation = Strategy([], [], time_metrics_trade=[TimeMetric(callables, 60)])
    backtester = backtest.Backtest(reader, simulation)

    row = reader.__next__()

    self.assertEqual((row.timestamp - simulation.time_metrics['trade'][0]._from).seconds, 0)



  def test_trades_len_minute_metric(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=1000)
    callables = [('trades count', lambda trades: len(trades))]
    simulation = Strategy([], [], time_metrics_trade=[TimeMetric(callables, 60)])

    output = TestOutput([], simulation.time_metrics['trade'][0].metric_names)
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.time_metrics), 87)

  def test_trades_volume_minute_metric(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=10000, pairs_to_load=10)
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    simulation = Strategy([], [], time_metrics_trade=[TimeMetric(callables, 60)])
    instant_metric_names = []

    output = TestOutput(instant_metric_names, simulation.time_metrics['trade'][0].metric_names)

    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.time_metrics), 2020)


  def test_all_metrics(self):
    reader = readers.SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=10000, pairs_to_load=5)
    # todo: optimize return metrics (do not waste time on wrapping each -> transform into tuple of values with one header
    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]

    metrics = [VWAP_volume(volumes=[100000, 1000000], symbol='XBTUSD'),
               VWAP_volume(volumes=[50000, 500000], symbol='ETHUSD')]
    simulation = Strategy(metrics, [Filters.DepthFilter(3)], time_metrics_trade=[TimeMetric(callables, 60)])
    instant_metric_names = [metric.names() for metric in metrics]
    output = TestOutput(instant_metric_names, simulation.time_metrics['trade'][0].metric_names)
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    self.assertEqual(len(output.instant_metrics), 46336)

  def test_load_orderbooks_and_floats(self):
    reader = readers.OrderbookReader('resources/orderbook10/orderbook.csv.gz',
                                     'resources/orderbook10/trades.csv.gz',
                                     pairs_to_load=5)

    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]

    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    time_metrics = [TimeMetric(callables, 60), TimeMetric(callables, 30)]

    simulation = Strategy(instant_metrics, time_metrics_trade=time_metrics)
    output = TestOutput(instant_metric_names=instant_metric_names,
                        time_metric_names=[metric.metric_names for metric in time_metrics])
    backtester = backtest.Backtest(reader, simulation, output)

    backtester.run()

    self.assertEqual(True, True)

  def test_order_simulation(self):
    # reader = readers.OrderbookReader('resources/orderbook10/orderbook.csv.gz',
    #                                  'resources/orderbook10/trades.csv.gz',
    #                                  pairs_to_load=5)

    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    time_metrics = [TimeMetric(callables, 60), TimeMetric(callables, 30)]
    simulation = Strategy(instant_metrics, time_metrics_trade=time_metrics)

    output = TestOutput(instant_metric_names=instant_metric_names,
                        time_metric_names=[metric.metric_names for metric in time_metrics])


    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 200),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300), 'Sell', 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300),
                np.array([9.5, 9.0, 8.5]), np.array([900, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 400),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400),
                np.array([9.0, 8.5, 8.0]), np.array([100, 100, 140]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 2000),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 1300),
    ])

    trigged = []
    simulation.trigger_trade = lambda *x: trigged.append(x)


    backtester = backtest.Backtest(reader, simulation, output)

    backtester._process_event(reader[0])


    # Sell это bid

    # add order request
    backtester._process_actions([OrderRequest.create_bid(9.5, 450, 'test', reader[0].timestamp)])
    # monitor request
    order_requests = backtester.simulated_orders[('test', 'bid')][9.5]
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(id, 0)
    self.assertEqual(volume_left, 1000)
    self.assertEqual(consumed, 0.0)

    # Check after trade event
    backtester._process_event(reader[1])
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 900)
    self.assertEqual(consumed, 0.0)


    for event in reader[2:4]:
      backtester._process_event(event)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 500)
    self.assertEqual(consumed, 0.0)

    backtester._process_event(reader[4])
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 0)
    self.assertEqual(consumed, 0.0)

    for event in reader[5:7]:
      backtester._process_event(event)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 0)
    self.assertAlmostEqual(consumed, 0.6666, delta=1e-3)

    backtester._process_event(reader[7])
    statuses: List[OrderStatus] = trigged[-1][1]
    status = statuses[0]
    self.assertEqual(status.at, reader[-1].timestamp)
    self.assertEqual(status.status, 'finished')

  def test_2order_simulation(self):
    # reader = readers.OrderbookReader('resources/orderbook10/orderbook.csv.gz',
    #                                  'resources/orderbook10/trades.csv.gz',
    #                                  pairs_to_load=5)

    callables = [
      ('trades volume', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('trades length', lambda trades: len(trades))
    ]
    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    time_metrics = [TimeMetric(callables, 60), TimeMetric(callables, 30)]
    simulation = Strategy(instant_metrics, time_metrics_trade=time_metrics)

    output = TestOutput(instant_metric_names=instant_metric_names,
                        time_metric_names=[metric.metric_names for metric in time_metrics])


    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 200),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300), 'Sell', 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300),
                np.array([9.5, 9.0, 8.5]), np.array([900, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 400),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), 'Sell', 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400),
                np.array([9.0, 8.5, 8.0]), np.array([100, 100, 140]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 2000),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), 'Sell', 9.0, 1300),
    ])

    trigged = []
    simulation.trigger_trade = lambda *x: trigged.append(x)


    backtester = backtest.Backtest(reader, simulation, output)

    backtester._process_event(reader[0])


    # Sell это bid

    # add order request
    backtester._process_actions([OrderRequest.create_bid(9.5, 450, 'test', reader[0].timestamp)])
    # monitor request
    order_requests = backtester.simulated_orders[('test', 'bid')][9.5]
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(id, 0)
    self.assertEqual(volume_left, 1000)
    self.assertEqual(consumed, 0.0)

    # Check after trade event
    backtester._process_event(reader[1])
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 900)
    self.assertEqual(consumed, 0.0)


    for event in reader[2:4]:
      backtester._process_event(event)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 500)
    self.assertEqual(consumed, 0.0)

    backtester._process_actions([OrderRequest.create_bid(9.5, 200, 'test', reader[3].timestamp)])

    backtester._process_event(reader[4])
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 0)
    self.assertEqual(consumed, 0.0)

    for event in reader[5:7]:
      backtester._process_event(event)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 0)
    self.assertAlmostEqual(consumed, 0.6666, delta=1e-3)

    backtester._process_event(reader[7])
    statuses: List[OrderStatus] = trigged[-1][1]
    status = statuses[0]
    self.assertEqual(status.at, reader[-1].timestamp)
    self.assertEqual(status.status, 'finished')

    second_order = backtester.simulated_orders_id[1]
    symbol, side, price = second_order.label()
    second_order = backtester.simulated_orders[(symbol, side)][price][0]
    consumed = second_order[2][0]
    self.assertAlmostEqual(consumed, 0.8625, delta=1e-3)
