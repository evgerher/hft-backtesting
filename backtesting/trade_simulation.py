from dataclasses import dataclass
from typing import List
from metrics.metrics import Metric
from metrics.filters import Filters

@dataclass
class Order:
  price: float
  volume: int
  side: str

class Simulation:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, metrics: List[Metric], filters: List[Filters.Filter]):
    self.metrics = metrics
    self.filters = filters

  def trigger(self):
    pass


class SimulationWithTrades(Simulation):
  pass  # todo: consider placed trade and trades arrival

class SimulationWithSnapshot(Simulation):
  pass  # todo: consider only snapshots deltas