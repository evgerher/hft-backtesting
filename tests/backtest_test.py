import datetime
import unittest
from typing import List

import numpy as np

from hft.backtesting import readers, backtest
from hft.backtesting.data import OrderRequest, OrderStatus
from hft.backtesting.output import StorageOutput
from hft.backtesting.readers import ListReader
from hft.backtesting.strategy import CalmStrategy
from hft.units.metrics.instant import VWAP_volume, VWAP_depth
from hft.units.metrics.time import TradeMetric
from hft.utils.consts import TradeSides, Statuses, QuoteSides
from hft.utils.data import OrderBook, Trade


class BacktestTest(unittest.TestCase):
  def test_run(self):
    reader = readers.OrderbookReader('resources/orderbook/_orderbooks.csv.gz', stop_after=100)
    metrics = [VWAP_depth(3),
               VWAP_volume(volumes=[int(5e5), int(1e6)], symbol='XBTUSD'),
               VWAP_volume(volumes=[int(1e5), int(5e5)], symbol='ETHUSD')]
    simulation = CalmStrategy(metrics)
    backtester = backtest.Backtest(reader, simulation, None)

    backtester.run()
    self.assertIsNotNone(backtester.memory.get(('orderbook','XBTUSD'), None))

    # self.assertTrue(len(backtester.units["('XBTUSD', 'VWAP (Depth): 3 bid')"]) > 2)

  def test_init_moment(self):
    reader = readers.OrderbookReader('resources/orderbook/_orderbooks.csv.gz')
    callables = [('_trades count', lambda trades: len(trades))]
    simulation = CalmStrategy([], time_metrics_trade=[TradeMetric(callables, 60)])
    backtester = backtest.Backtest(reader, simulation)

    obj, boolean = reader.__next__()

    self.assertEqual((obj.timestamp - simulation.time_metrics['trade'][0]._from).seconds, 0)

  def test_trades_volume_minute_metric(self):
    reader = readers.OrderbookReader('resources/orderbook/_orderbooks.csv.gz',
                                     trades_file='resources/orderbook/_trades.csv.gz', stop_after=10000, depth_to_load=10)
    callables = [
      ('_trades volume_total', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('_trades length', lambda trades: len(trades))
    ]
    simulation = CalmStrategy([], time_metrics_trade=[TradeMetric(callables, 60)])
    instant_metric_names = []

    output = StorageOutput(instant_metric_names, simulation.time_metrics['trade'][0].metric_names)

    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    time_metrics_q = sum([len(item) for item in output.time_metrics.values()])
    self.assertEqual(time_metrics_q, len(output.trades))
    self.assertEqual(time_metrics_q, 715)

  def test_all_metrics(self):
    reader = readers.OrderbookReader('resources/orderbook/_orderbooks.csv.gz', trades_file='resources/orderbook/_trades.csv.gz', stop_after=3000, depth_to_load=6)
    callables = [
      ('_trades volume_total', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('_trades length', lambda trades: len(trades))
    ]

    metrics = [VWAP_volume(volumes=[100000, 1000000], symbol='XBTUSD'),
               VWAP_volume(volumes=[50000, 500000], symbol='ETHUSD')]
    simulation = CalmStrategy(metrics, time_metrics_trade=[TradeMetric(callables, 60)])
    instant_metric_names = [metric.names() for metric in metrics]
    output = StorageOutput(instant_metric_names, simulation.time_metrics['trade'][0].metric_names)
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()
    instant_metrics_q = sum([len(item) for item in output.instant_metrics.values()])
    time_metrics_q = sum([len(item) for item in output.time_metrics.values()])
    self.assertEqual(instant_metrics_q, 4748)
    self.assertEqual(time_metrics_q, 283)
    # todo: FIX THIS, 2 is incorrect

  def test_load_orderbooks_and_floats(self):
    reader = readers.OrderbookReader('resources/orderbook/_orderbooks.csv.gz',
                                     'resources/orderbook/_trades.csv.gz',
                                     depth_to_load=5, stop_after=500)

    callables = [
      ('_trades volume_total', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('_trades length', lambda trades: len(trades))
    ]

    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    time_metrics = [TradeMetric(callables, 60), TradeMetric(callables, 30)]

    simulation = CalmStrategy(instant_metrics, time_metrics_trade=time_metrics)
    output = StorageOutput(instant_metric_names=instant_metric_names,
                           time_metric_names=[metric.metric_names for metric in time_metrics])
    backtester = backtest.Backtest(reader, simulation, output)
    backtester.run()

    instant_metrics_q = sum([len(item) for item in output.instant_metrics.values()])
    self.assertEqual(instant_metrics_q, 239+139)

  def test_2order_simulation(self):
    # reader = readers.OrderbookReader('resources/orderbook10/orderbook.csv.gz',
    #                                  'resources/orderbook10/_trades.csv.gz',
    #                                  pairs_to_load=5)

    callables = [
      ('_trades volume_total', lambda trades: sum(map(lambda x: x.volume, trades))),
      ('_trades length', lambda trades: len(trades))
    ]
    instant_metrics = [
      VWAP_volume(volumes=[50000, 500000])
    ]
    instant_metric_names = [metric.names() for metric in instant_metrics]

    time_metrics = [TradeMetric(callables, 60), TradeMetric(callables, 30)]
    simulation = CalmStrategy(instant_metrics, time_metrics_trade=time_metrics)

    output = StorageOutput(instant_metric_names=instant_metric_names,
                           time_metric_names=[metric.metric_names for metric in time_metrics])


    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 200),
                np.array([9.5, 9.0, 8.5]), np.array([1100, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300), TradeSides.SELL, 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 300),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), TradeSides.SELL, 9.5, 400),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400), TradeSides.SELL, 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 400),
                np.array([9.5, 9.0, 8.5]), np.array([100, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), TradeSides.SELL, 9.0, 325),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, 500), TradeSides.SELL, 9.0, 400),
    ])

    trigged = []

    initial_call = simulation.trigger
    def substitute_action(*args):
      res = initial_call(*args)
      trigged.append(args)
      return res

    simulation.trigger = substitute_action


    backtester = backtest.Backtest(reader, simulation, output)

    backtester._process_event(reader[0], True)


    # Sell это bid

    # add order request
    backtester._process_actions([OrderRequest.create_bid(9.5, 450, 'test', reader[0].timestamp)])
    # monitor request
    order_requests = backtester.simulated_orders[('test', QuoteSides.BID)][9.5]
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(id, 0)
    self.assertEqual(volume_left, 1100)
    self.assertEqual(consumed, 0.0)

    # Check after trade event
    backtester._process_event(reader[1], False)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 1000)
    self.assertEqual(consumed, 0.0)


    for event in reader[2:4]:
      backtester._process_event(event, type(event) == OrderBook)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 600)
    self.assertEqual(consumed, 0.0)


    backtester._process_event(reader[4], type(reader[4]) == OrderBook)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 100)
    self.assertEqual(consumed, 0.0)

    backtester._process_event(reader[5], type(reader[5]) == OrderBook)
    backtester._process_actions([OrderRequest.create_bid(9.5, 200, 'test', reader[3].timestamp)])
    second_order = backtester.simulated_orders[('test', QuoteSides.BID)][9.5][1]
    self.assertEqual(550, second_order[1])

    backtester._process_event(reader[6], type(reader[6]) == OrderBook)
    id, volume_left, consumed = order_requests[0]
    self.assertEqual(volume_left, 0)
    self.assertAlmostEqual(consumed, 0.5, delta=1e-3)

    backtester._process_event(reader[7], type(reader[7]) == OrderBook)
    statuses: List[OrderStatus] = trigged[-1][1]
    status = statuses[0]
    self.assertEqual(status.at, reader[-1].timestamp)
    self.assertEqual(status.status, Statuses.FINISHED)

    second_order = backtester.simulated_orders_id[1]
    symbol, side, price = second_order.label()
    second_order = backtester.simulated_orders[(symbol, side)][price][0]
    consumed = second_order[2]
    self.assertAlmostEqual(consumed, 175./200, delta=1e-3)

  def test_delay(self):
    simulation = CalmStrategy()

    output = StorageOutput([], [])

    reader = ListReader([
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=200000),
                np.array([9.5, 9.0, 8.5]), np.array([1100, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=300000), TradeSides.SELL, 9.5, 100),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=300000),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=350000), TradeSides.SELL, 9.5, 1150),
      Trade('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=350000), TradeSides.SELL, 9.5, 500),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=350000),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 500])),
      OrderBook('test', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=500000),
                np.array([9.5, 9.0, 8.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 600]))
    ])

    simulation.trigger = lambda *args: [] # disable inner logic for strategy
    backtester = backtest.Backtest(reader, simulation, output, delay=100)

    backtester._process_event(reader[0], type(reader[0]) == OrderBook)
    backtester._process_actions([OrderRequest.create_bid(9.5, 450, 'test', reader[0].timestamp)])
    self.assertEqual(len(backtester.simulated_orders), 0)
    self.assertEqual(len(backtester.simulated_orders_id), 0)
    self.assertEqual(len(backtester.pending_orders), 1)


    backtester._process_event(reader[1], type(reader[1]) == OrderBook)
    backtester._process_event(reader[2], type(reader[2]) == OrderBook)
    self.assertEqual(len(backtester.simulated_orders[('test', QuoteSides.BID)][9.5]), 1)
    self.assertEqual(len(backtester.simulated_orders_id), 1)
    self.assertEqual(len(backtester.pending_orders), 0)
    self.assertEqual(len(backtester.pending_statuses), 0)

    backtester._process_event(reader[3], type(reader[3]) == OrderBook)
    self.assertEqual(len(backtester.simulated_orders[('test', QuoteSides.BID)][9.5]), 1)
    self.assertEqual(len(backtester.simulated_orders_id), 1)
    self.assertEqual(len(backtester.pending_statuses), 1)

    backtester._process_event(reader[4], type(reader[4]) == OrderBook)
    self.assertEqual(len(backtester.simulated_orders[('test', QuoteSides.BID)][9.5]), 0)
    self.assertEqual(len(backtester.simulated_orders_id), 0)
    self.assertEqual(len(backtester.pending_statuses), 2)

    backtester._process_event(reader[5], type(reader[5]) == OrderBook)
    backtester._process_event(reader[6], type(reader[6]) == OrderBook)
    self.assertEqual(len(backtester.pending_statuses), 0)

  @unittest.skip("Works only alone")
  def test_cancel(self):
    simulation = CalmStrategy()
    output = StorageOutput([], [])
    reader = ListReader([
      OrderBook('XBTUSD', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=200000),
                np.array([9.5, 9.0, 8.5]), np.array([1100, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),
      OrderBook('XBTUSD', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=300000),
                np.array([10.0, 9.5, 9.0]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),

      OrderBook('XBTUSD', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=310000),
                np.array([10.5, 10.0, 9.5]), np.array([1000, 100, 100]),
                np.array([10.0, 10.5, 11.0]), np.array([100, 100, 100])),

      OrderBook('XBTUSD', datetime.datetime(2020, 3, 10, 8, 10, 30, microsecond=350000),
                np.array([10.5, 10.0, 9.5]), np.array([1000, 100, 100]),
                np.array([8.5, 9.0, 9.5]), np.array([100, 100, 500])),
    ])

    backtester = backtest.Backtest(reader, simulation, output, delay=0)

    backtester._process_event(reader[0], type(reader[0]) == OrderBook)
    backtester._process_actions([OrderRequest.create_bid(9.5, 200, 'XBTUSD', reader[0].timestamp)])
    backtester._process_actions([OrderRequest.create_ask(10.0, 200, 'XBTUSD', reader[0].timestamp)])
    self.assertEqual(len(backtester.simulated_orders_id), 2)

    backtester._process_event(reader[1], type(reader[1]) == OrderBook)
    self.assertEqual(len(backtester.simulated_orders_id), 2)

    backtester._process_event(reader[2], type(reader[2]) == OrderBook)
    self.assertEqual(len(backtester.simulated_orders_id), 1)

    backtester._process_event(reader[3], type(reader[3]) == OrderBook)
    self.assertEqual(len(backtester.simulated_orders_id), 0)


if __name__ == '__main__':
  unittest.main()
