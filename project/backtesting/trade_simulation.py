import datetime
from dataclasses import dataclass
from typing import List, Deque, Dict

from utils.data import Snapshot
from metrics.metrics import MetricEvaluator, Metric
from metrics.filters import Filters

@dataclass
class Order:
  price: float
  volume: int
  side: str

class Simulation:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, metric_evaluators: List[MetricEvaluator], filters: List[Filters.Filter], delay = 400e-6):
    """

    :param metric_evaluators:
    :param filters:
    :param delay:
    """
    self.metric_evaluators = metric_evaluators
    self.filters = filters
    self.delay = delay

  def trigger(self, row: Snapshot,
              memory: Dict[str, Deque[Snapshot]],
              metrics: Dict[(str, str), Deque[(datetime.datetime.timestamp, Metric)]],
              trades): # todo: define type later on
    pass


class SimulationWithTrades(Simulation):
  pass  # todo: consider placed trade and trades arrival

class SimulationWithSnapshot(Simulation):
  pass  # todo: consider only snapshots deltas