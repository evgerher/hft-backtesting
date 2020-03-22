import datetime
from collections import defaultdict

from utils.data import OrderBook, Trade
from abc import ABC, abstractmethod


class Output(ABC):
  def __init__(self, instant_metric_names=None, time_metric_names=None):
    self.instant_metric_names = instant_metric_names
    self.time_metric_names = time_metric_names

  def consume(self, labels, timestamp: datetime.datetime, object):
    """

    :param timestamp:
    :param object:
    :return:
    """

    if 'trade' in labels:
      self.trade_action(timestamp, object)
    elif 'snapshot' in labels:
      self.snapshot_action(timestamp, object)
    elif 'trade-time-metric' in labels:
      self.time_metric_action(timestamp, labels, object)
    elif 'snapshot-instant-metric' in labels:
      self.instant_metric_action(timestamp, labels, object)
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


class TestOutput(Output):
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
