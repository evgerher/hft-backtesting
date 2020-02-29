from backtesting.output import Output
from backtesting.readers import Reader
from backtesting.trade_simulation import Simulation
from utils.data import Snapshot, Trade
from utils.logger import setup_logger
from metrics.metrics import MetricData, TimeMetric

import datetime
from typing import Dict, List, Deque, Optional, Tuple
from collections import defaultdict, deque


logger = setup_logger('<backtest>', 'DEBUG')


class Backtest:

  def __init__(self, reader: Reader,
               simulation: Simulation,
               output: Optional[Output] = None,
               time_horizon:int=360):
    """

    :param reader:
    :param simulation:
    :param time_horizon: time in seconds for storaging Snapshots ; todo: how to work with trades?
    """
    self.reader : Reader = reader
    self.simulation: Simulation = simulation
    self.time_horizon: int = time_horizon
    self.memory: Dict[str, Deque[Snapshot]] = defaultdict(deque)
    self.metrics: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, MetricData]]] = defaultdict(deque)  # (symbol, metric_name) -> (timestamp, Metric)
    self.trades: Dict[Tuple[str, str], Deque[Trade]] = defaultdict(deque)
    self.output = output

    self.__initialize_time_metrics()

    logger.info(f"Initialized {self}")

  def run(self):
    def _filter_snapshot(row: Snapshot) -> bool:
      filtered = True
      for filter in self.simulation.filters:
        if not filter.filter(row):
          filtered = False
          break

      return filtered

    logger.info(f'Backtest initialize run')
    for row in self.reader:
      if type(row) is Snapshot:
        if not _filter_snapshot(row):
          continue
        self._update_memory(row)
        self._update_metrics(row)  # todo: what to do with pair FX features?
        # todo: what if I put a trade?
        actions = self.simulation.trigger(row, self.memory, self.metrics, self.trades)
        if actions is not None:
          self._process_actions(actions)
      elif type(row) is Trade:
        self._update_trades(row)

    self._flush_last()
    logger.info(f'Backtest finished run')

  def __initialize_time_metrics(self):
    for metric in self.simulation.time_metrics:
      metric.set_starting_moment(self.reader.initial_moment)

  def _flush_output(self, timestamp: datetime.datetime, object): # todo: mark as Generic
    """

    :param timestamp:
    :param object: may be Metric/Snapshot/Trade
    :return:
    """
    if self.output is not None:
      self.output.consume(timestamp, object)

  # todo: refactor _update_* into one function
  def _update_memory(self, row: Snapshot):
    logger.debug(f'Update memory with snapshot symbol={row.symbol} @ {row.timestamp}')
    # # fill memory
    market: Deque[Snapshot] = self.memory[row.symbol]
    market.append(row)

    self.__remove_old_memory(row.timestamp, market)

  def _update_trades(self, row: Trade):

    logger.debug(f'Update memory with trade symbol={row.symbol}, side={row.side} @ {row.timestamp}')
    market: Deque[Trade] = self.trades[(row.symbol, row.side)]
    market.append(row)
    self.__remove_old_memory(row.timestamp, market)

    for time_metric in self.simulation.time_metrics:

      values: List[MetricData] = time_metric.evaluate(row)
      for value in values:
        metric_name = (row.symbol, value.name)
        metric_deque: Deque[(datetime.datetime, MetricData)] = self.metrics[metric_name]
        metric_deque.append((row.timestamp, value))

        self.__remove_old_metric(row, metric_deque)

  def _update_metrics(self, row: Snapshot):
    logger.debug(f'Update metrics with snapshot symbol={row.symbol} @ {row.timestamp}')

    for instant_metric in self.simulation.instant_metrics:
      values: List[MetricData] = instant_metric.evaluate(row)

      for value in values:
        metric_name = (row.symbol, value.name)
        metric_deque: Deque[(datetime.datetime, MetricData)] = self.metrics[metric_name]
        metric_deque.append((row.timestamp, value))

        self.__remove_old_metric(row, metric_deque)

  def _process_actions(self, actions: List):
    pass

  def _flush_last(self):
    """

    :return: flush contents of metric storage values when dataset is finished
    """
    for metric in self.metrics.values():
      while len(metric) > 0:
        self._flush_output(*metric.popleft())

  def __remove_old_memory(self, timestamp: datetime.datetime, metric_deque: Deque[Trade]):
    while True:
      if (timestamp - metric_deque[0].timestamp).seconds > self.time_horizon:
        object = metric_deque.popleft()
        self._flush_output(object.timestamp, object)
      else:
        break

  def __remove_old_metric(self, row, metric_deque):
    while True:
      if (row.timestamp - metric_deque[0][0]).seconds > self.time_horizon:
        self._flush_output(*metric_deque.popleft())
      else:
        break

  def __str__(self):
    return '<Backtest with reader={}>'.format(self.reader)
