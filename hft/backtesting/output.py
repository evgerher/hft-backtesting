import datetime
from collections import defaultdict
from typing import Sequence, Optional, List, Tuple
from abc import ABC, abstractmethod

from hft.backtesting.data import OrderRequest
from hft.utils.consts import QuoteSides
from hft.utils.data import OrderBook, Trade
from hft.utils import helper

import pandas as pd
import matplotlib.pyplot as plt

class Output(ABC):

  @abstractmethod
  def consume(self, labels, timestamp: datetime.datetime, object):
    raise NotImplementedError

class OutputLabeled(Output, ABC):
  def __init__(self, instant_metric_names=None, time_metric_names=None):
    self.instant_metric_names = instant_metric_names
    self.time_metric_names = time_metric_names

  def consume(self, labels, timestamp: datetime.datetime, object):
    """

    :param timestamp:
    :param object:
    :return:
    """

    if type(object) == Trade:
      self.trade_action(timestamp, object)
    elif type(object) == OrderBook:
      self.snapshot_action(timestamp, object)
    elif 'instant-metric' in labels:
      self.instant_metric_action(timestamp, labels, object)
    elif 'time-metric' in labels:
      self.time_metric_action(timestamp, labels, object)
    else:
      self.additional_action(timestamp, labels, object)

  @abstractmethod
  def snapshot_action(self, timestamp: datetime.datetime, object):
    raise NotImplementedError

  @abstractmethod
  def metric_action(self, timestamp: datetime.datetime, object):
    raise NotImplementedError

  @abstractmethod
  def trade_action(self, timestamp: datetime.datetime, object: Trade):
    raise NotImplementedError

  @abstractmethod
  def additional_action(self, timestamp: datetime.datetime, labels, object):
    raise NotImplementedError

  @abstractmethod
  def instant_metric_action(self, timestamp, labels, object):
    raise NotImplementedError

  @abstractmethod
  def time_metric_action(self, timestamp, labels, object):
    raise NotImplementedError


class StorageOutput(OutputLabeled):
  # todo: make it return list of accessible fields, like model.describe()
  def __init__(self, instant_metric_names, time_metric_names):
    super().__init__(instant_metric_names, time_metric_names)
    self.snapshots = []
    self.instant_metrics = defaultdict(list)
    self.time_metrics = defaultdict(list)
    self.others = []
    self.trades = []

  def time_metric_action(self, timestamp, labels, object):
    self.time_metrics[tuple(labels)].append((timestamp, object))

  def snapshot_action(self, timestamp: datetime.datetime, object: OrderBook):
    self.snapshots.append((timestamp, object))

  def instant_metric_action(self, timestamp, labels, object):
    self.instant_metrics[tuple(labels)].append((timestamp, object))

  def trade_action(self, timestamp: datetime.datetime, object: Trade):
    self.trades.append((timestamp, object))

  def additional_action(self, timestamp: datetime.datetime, labels, object):
    self.others.append((timestamp, labels, object))

  def metric_action(self, timestamp: datetime.datetime, object):
    pass


class SimulatedOrdersOutput(Output):
  def __init__(self):
    self.orders = defaultdict(list)

  def consume(self, labels, timestamp: datetime.datetime, object):
    if 'order-request' in labels and labels[-1] is not None: # todo: bad fix for cancel removals
      self.orders[tuple(labels)].append(object)


def make_plot_orderbook_trade(orderbook_file: str, symbol: str,
                              simulated_orders: Optional[Sequence[OrderRequest]] = None,
                              no_action_ts: Optional[List[Tuple[datetime.datetime, float]]] = None,
                              orderbook_precomputed:bool=False,
                              figsize=(16,6),
                              skip_every=20,
                              savefig=False) -> Tuple[plt.Figure, plt.Axes]:
  '''
  Utility function, reads file and plots price of orderbook
  If simulated orders are provided, scatter them on a plot with orderbook prices
  # If trade_file is provided, scatter them on a plot with orderbook prices

  :param orderbook_file: to read
  :param symbol: to filter by
  :param simulated_orders: to display among with ob prices
  :param orderbook_precomputed: flag whether millis are already evaluated
  :param figsize:
  :param skip_every: every i-th snapshot's price will be present on a plot
  :return: plot tuple (figure, axis)
  '''
  import matplotlib.ticker as ticker

  orderbooks = pd.read_csv(orderbook_file, header=None)
  if not orderbook_precomputed:
    orderbooks = helper.fix_timestamp(orderbooks, 0, 1, orderbook_precomputed)

  orderbooks = orderbooks[orderbooks[2] == symbol]
  orderbooks = orderbooks.iloc[::skip_every, :]
  ts = pd.to_datetime(orderbooks[0])
  best_bid_prices = orderbooks[23]
  best_ask_prices = orderbooks[3]

  fig, axs = plt.subplots(figsize=figsize)
  axs.plot(ts, best_bid_prices, c='r', label='bid prices')
  axs.plot(ts, best_ask_prices, c='b', label='ask prices')

  if simulated_orders is not None:
    orders = filter(lambda x: x.symbol == symbol, simulated_orders)
    sides = {QuoteSides.BID: [], QuoteSides.ASK: []}
    for order in orders:
      sides[order.side].append((order.created, order.price))

    tss, prices = zip(*sides[QuoteSides.BID])
    axs.scatter(tss, prices, c='y', label='Simulated bid orders')

    tss, prices = zip(*sides[QuoteSides.ASK])
    axs.scatter(tss, prices, c='g', label='Simulated ask orders')

  if no_action_ts is not None and len(no_action_ts) > 0:
    tss, prices = zip(*no_action_ts)
    axs.scatter(tss, prices, c='yellow', label='No action applied')

  axs.xaxis.set_major_locator(ticker.AutoLocator())
  plt.legend()
  plt.xticks(rotation=90)
  if savefig:
    fname = orderbook_file.rsplit("/")[-1]
    plt.savefig(f'{fname}.png')
  return fig, axs
