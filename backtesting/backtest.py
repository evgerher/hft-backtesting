from backtesting.readers import Reader
from backtesting.trade_simulation import Simulation
from dataloader.utils.data import Snapshot
from metrics.metrics import Metric

import datetime
from typing import Dict, List, Deque
from collections import defaultdict, deque


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
    self.memory: Dict[str, Deque[Snapshot]] = defaultdict(deque)
    self.metrics: Dict[(str, str), Deque[(datetime.datetime.timestamp, Metric)]] = defaultdict(deque)
    self.trades = None
    self.output = output

  def flush_output(self, timestamp: datetime.datetime.timestamp, metric: Metric):
    if self.output is not None:
      self.output((timestamp, metric))

  def _update_memory(self, row: Snapshot):
    # # fill memory
    market: Deque[Snapshot] = self.memory[row.market]
    market.append(row)

    # todo: test, not sure it will work (references, blah, blah)
    while True:
      if (row.timestamp - market[0].timestamp).seconds > self.time_horizon:
        market.popleft()  # todo: pop it somewhere into output channel
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
    for metric in self.simulation.metrics:
      values: List[Metric] = metric.evaluate(row)

      for value in values:
        metric_name = (row.market, value.name)
        metric_deque: Deque[(datetime.datetime.timestamp, Metric)] = self.metrics[metric_name]
        metric_deque.append((row.timestamp, value))

        while True:
          if (row.timestamp - metric_deque[0]).seconds > self.time_horizon:
            self.flush_output(*metric_deque.popleft())
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
      # self.simulation.trigger(row, self.memory, self.metrics, self.trades)
