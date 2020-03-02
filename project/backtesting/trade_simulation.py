import datetime
from dataclasses import dataclass
from typing import List, Deque, Dict, Tuple, Optional

from utils.data import Snapshot
from metrics.metrics import InstantMetric, MetricData, TimeMetric
from metrics.filters import Filters

@dataclass
class Order:
  price: float
  volume: int
  side: str

class Simulation:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, instant_metrics: List[InstantMetric],
               filters: List[Filters.Filter], time_metrics: Optional[List[TimeMetric]] = None, delay = 400e-6):
    """

    :param instant_metrics:
    :param filters:
    :param delay:
    """
    self.instant_metrics: List[InstantMetric] = instant_metrics
    self.filters: List[Filters.Filter] = filters
    self.time_metrics: List[TimeMetric] = time_metrics if time_metrics is not None else []
    self._delay: int = delay


  def trigger(self, row: Snapshot,
              memory: Dict[str, Deque[Snapshot]],
              metrics: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, MetricData]]],
              trades) -> List: # todo: define type later on
    pass


class SimulationWithTrades(Simulation):
  pass  # todo: consider placed trade and trades arrival

class SimulationWithSnapshot(Simulation):
  pass  # todo: consider only snapshots deltas