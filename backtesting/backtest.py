from backtesting.readers import Reader
from backtesting.trade_simulation import Simulation
from dataloader.utils.data import Snapshot
from metrics.metrics import Metric

import datetime
from typing import Dict, List
from collections import defaultdict


class Backtest:

  def __init__(self, reader: Reader, simulation: Simulation, output: None, time_horizon=360):
    """

    :param reader:
    :param simulation:
    :param time_horizon: time in seconds for storaging Snapshots ; todo: how to work with trades?
    """
    self.reader = reader
    self.simulation = simulation
    self.time_horizon = time_horizon
    self.memory: Dict[str, List[Snapshot]] = defaultdict(list)
    self.metrics: Dict[(str, str), List[(datetime.datetime.timestamp, float)]] = defaultdict(list)

  def _update_memory(self, row: Snapshot):
    # # fill memory
    market: List[Snapshot] = self.memory[row.market]
    market.append(row)

    # todo: test, not sure it will work (references, blah, blah)
    while True:
      if (market[-1].timestamp - market[0].timestamp).seconds > self.time_horizon:
        market.pop(0)  # todo: pop it somewhere into output channel
      else:
        break

  def _filter(self, row: Snapshot) -> bool:
    filtered = True
    for filter in self.simulation.filters:
      if not filter.filter(row):
        filtered = False
        break

    return filtered

  def _update_metrics(self, row: Snapshot):
    current_t = row.timestamp
    for metric in self.simulation.metrics:
      values: List[Metric] = metric.evaluate(row) # todo: does not work

      for value in values:
        metric_name = (row.market, value.name)
        self.metrics[metric_name].append((current_t, value.value))

        while True:
          if (current_t - self.metrics[metric_name][0]).seconds > self.time_horizon:
            self.metrics[metric_name].pop(0) # todo: pop it somewhere into output channel
          else:
            break

  def run(self):
    while self.reader.has_next():
      row: Snapshot = self.reader.read_next()  # todo: how to work with trades?

      if not self._filter(row):
        continue

      self._update_memory(row)
      self._update_metrics(row) # todo: what to do with pair FX features?
      # todo: what if I put a trade?
      # self.simulation.trigger(row, self.memory, self.metrics)
