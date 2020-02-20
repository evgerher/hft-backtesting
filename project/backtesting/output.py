import datetime

from utils.data import Snapshot
from metrics.metrics import Metric


class Output:
  def consume(self, timestamp: datetime.datetime.timestamp, object):
    """

    :param timestamp:
    :param object:
    :return:
    """

    if type(object) == Snapshot:
      self.snapshot_action(timestamp, object)

    if type(object) == Metric:
      self.metric_action(timestamp, object)

    self.additional_action(timestamp, object)

  def snapshot_action(self, timestamp, object):
    pass

  def metric_action(self, timestamp, object):
    pass

  def additional_action(self, timestamp, object):
    pass
