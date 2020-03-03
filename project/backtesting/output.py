import datetime

from utils.data import OrderBook
from metrics.metrics import MetricData


class Output:
  def consume(self, timestamp: datetime.datetime, object):
    """

    :param timestamp:
    :param object:
    :return:
    """

    if isinstance(object, OrderBook):
      self.snapshot_action(timestamp, object)

    if isinstance(object, MetricData):
      self.metric_action(timestamp, object)

    self.additional_action(timestamp, object)

  def snapshot_action(self, timestamp: datetime.datetime, object):
    pass

  def metric_action(self, timestamp: datetime.datetime, object):
    pass

  def additional_action(self, timestamp: datetime.datetime, object):
    pass


class TestOutput(Output):
  def __init__(self):
    self.snapshots = []
    self.metrics = []
    self.others = []

  def snapshot_action(self, timestamp: datetime.datetime, object):
    self.snapshots.append((timestamp, object))

  def metric_action(self, timestamp: datetime.datetime, object):
    self.metrics.append((timestamp, object))

  def additional_action(self, timestamp: datetime.datetime, object):
    self.others.append((timestamp, object))
