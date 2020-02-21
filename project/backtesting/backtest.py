from backtesting.output import Output
from backtesting.readers import Reader
from backtesting.trade_simulation import Simulation
from utils.data import Snapshot
from metrics.metrics import Metric

import datetime
from typing import Dict, List, Deque
from collections import defaultdict, deque


class Backtest:

  def __init__(self, reader: Reader, simulation: Simulation, output: Output = None, time_horizon=360):
    """

    :param reader:
    :param simulation:
    :param time_horizon: time in seconds for storaging Snapshots ; todo: how to work with trades?
    """
    self.reader = reader
    self.simulation = simulation
    self.time_horizon = time_horizon
    self.memory: Dict[str, Deque[Snapshot]] = defaultdict(deque)
    self.metrics: Dict[(str, str), Deque[(datetime.datetime.timestamp, Metric)]] = defaultdict(deque)  # (market, metric_name) -> (timestamp, Metric)
    self.trades = None  # Dict[(str, str), (datetime.datetime.timestamp, Trade)] = None  # todo: (market, side) -> (timestamp, trade)
    self.output = output.consume

  def _flush_output(self, timestamp: datetime.datetime.timestamp, object):
    """

    :param timestamp:
    :param object: may be Metric/Snapshot/Trade
    :return:
    """
    if self.output is not None:
      self.output(timestamp, object)

  def _update_memory(self, row: Snapshot):
    # # fill memory
    market: Deque[Snapshot] = self.memory[row.market]
    market.append(row)

    # todo: test, not sure it will work (references, blah, blah)
    while True:
      if (row.timestamp - market[0].timestamp).seconds > self.time_horizon:
        object = market.popleft()
        self._flush_output(object.timestamp, object)
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
    for metric_evaluator in self.simulation.metric_evaluators:
      values: List[Metric] = metric_evaluator.evaluate(row)

      for value in values:
        metric_name = (row.market, value.name)
        metric_deque: Deque[(datetime.datetime.timestamp, Metric)] = self.metrics[metric_name]
        metric_deque.append((row.timestamp, value))

        while True:
          if (row.timestamp - metric_deque[0]).seconds > self.time_horizon:
            self._flush_output(*metric_deque.popleft())
          else:
            break

  def _flush_last(self):
    """

    :return: flush contents of metric storage values when dataset is finished
    """
    for metric in self.metrics.values():
      while len(metric) > 0:
        self._flush_output(*metric.popleft())

  def run(self):
    for row in self.reader:
      # todo: how to work with trades?
      if not self._filter(row):
        continue
      self._update_memory(row)
      self._update_metrics(row) # todo: what to do with pair FX features?
      # todo: what if I put a trade?
      self.simulation.trigger(row, self.memory, self.metrics, self.trades)

    self._flush_last()