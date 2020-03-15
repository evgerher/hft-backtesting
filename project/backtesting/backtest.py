from backtesting.output import Output
from backtesting.readers import Reader
from backtesting.trade_simulation import Strategy, OrderRequest, OrderStatus
from utils.data import OrderBook, Trade
from utils.logger import setup_logger

import datetime
from typing import Dict, List, Deque, Optional, Tuple, Union
from collections import defaultdict, deque
import random


logger = setup_logger('<backtest>', 'INFO')


class Backtest:

  def __init__(self, reader: Reader,
               simulation: Strategy,
               output: Optional[Output] = None,
               order_position_policy: str = 'top', # 'random' or 'bottom'
               time_horizon:int=120,
               seed=1337):
    """

    :param reader:
    :param simulation:
    :param time_horizon: time in seconds for storaging Snapshots
    """
    self.reader: Reader = reader
    self.simulation: Strategy = simulation
    self.time_horizon: int = time_horizon

    self.memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, OrderBook]]] = defaultdict(deque)

    # (symbol) -> (timestamp, instant-metric-values)
    self.snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]] = defaultdict(deque)

    # (symbol, action, window-size) -> (timestamp, time-metric-values)
    self.trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]] = defaultdict(deque)

    # (symbol, action) -> Trade
    self.trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]] = defaultdict(deque)
    self.output: Output = output

    # (symbol, side, price) -> List[(order_id, position)]
    self.simulated_trades: Dict[Tuple[str, str, float], List[Tuple[int, float]]] = defaultdict(list)
    # id -> request
    self.simulated_trades_id: Dict[int, OrderRequest] = {}

    if order_position_policy == 'top':
      policy = lambda: 1.0
    elif order_position_policy == 'bottom':
      policy = lambda: 0.0
    elif order_position_policy == 'random':
      r = random.Random()
      r.seed(seed)
      policy = lambda: r.uniform(0.0, 1.0)
    else:
      policy = lambda: 1.0

    self._generate_initial_position = policy
    self.__initialize_time_metrics()

    logger.info(f"Initialized {self}")

  def run(self):
    def _filter_snapshot(row: OrderBook) -> bool:
      filtered = True
      for filter in self.simulation.filters:
        if not filter.filter(row):
          filtered = False
          break

      return filtered

    logger.info(f'Backtest initialize run')
    for row in self.reader:
      actions = None
      if type(row) is OrderBook:
        if not _filter_snapshot(row):
          continue
        # self._update_memory(row)
        self._update_metrics(row)
        actions = self.simulation.trigger_snapshot(row,
                                                   self.memory, self.snapshot_instant_metrics,
                                                   self.trade_time_metrics, self.trades)
      elif type(row) is Trade: # it must be cumulative trade if any
        self._update_trades(row)
        statuses = self._evaluate_statuses()
        actions = self.simulation.trigger_trade(row, statuses,
                                                self.memory, self.snapshot_instant_metrics,
                                                self.trade_time_metrics, self.trades)

      if actions is not None:
        self._process_actions(actions)

    # self._flush_last()
    logger.info(f'Backtest finished run')

  def _evaluate_statuses(self) -> List[OrderStatus]:
    pass

  def _process_actions(self, actions: List[OrderRequest]):
    for action in actions:
      if action.command == 'new':
        self.simulated_trades[action.label()] = (action.id, action.volume, self._generate_initial_position())
        self.simulated_trades_id[action.id] = action
      elif action.command == 'delete':
        order = self.simulated_trades_id.pop(action.id)
        del self.simulated_trades[order.label()]

  def __initialize_time_metrics(self):
    for metric in self.simulation.time_metrics:
      metric.set_starting_moment(self.reader.initial_moment)

  def _flush_output(self, labels: List[str], timestamp: datetime.datetime, values: List[float]):
    """

    :param timestamp:
    :param object: may be Metric/Snapshot/Trade
    :return:
    """
    if self.output is not None:
      self.output.consume(labels, timestamp, values)

  # todo: refactor _update_* into one function
  def _update_memory(self, row: OrderBook):
    logger.debug(f'Update memory with snapshot symbol={row.symbol} @ {row.timestamp}')
    # # fill memory
    market: Deque[Tuple[datetime.datetime, OrderBook]] = self.memory[(row.symbol, )]
    market.append((row.timestamp, row))

    self._flush_output(['snapshot'], row.timestamp, row)
    self.__remove_old_metric('snapshot', row.timestamp)

  def _update_trades(self, row: Trade):

    logger.debug(f'Update memory with trade symbol={row.symbol}, side={row.side} @ {row.timestamp}')
    market: Deque[Tuple[datetime.datetime, Trade]] = self.trades[(row.symbol, row.side)]
    market.append((row.timestamp, row))

    self._flush_output(['trade'], row.timestamp, row)
    self.__remove_old_metric('trade', row.timestamp)

    # update timemetrics
    values: List[float] = []
    for time_metric in self.simulation.time_metrics:
      values += time_metric.evaluate(row)

      metric_name = (row.symbol, row.side, time_metric.seconds)
      metric_deque: Deque[(datetime.datetime, List[float])] = self.trade_time_metrics[metric_name]
      metric_deque.append((row.timestamp, values))

      self._flush_output(['trade-time-metric', *metric_name], row.timestamp, values)
      self.__remove_old_metric('trade-time-metric', row.timestamp)

  def _update_metrics(self, row: OrderBook):
    logger.debug(f'Update metrics with snapshot symbol={row.symbol} @ {row.timestamp}')

    for instant_metric in self.simulation.instant_metrics:
      values = instant_metric.evaluate(row)

      metric_name = (row.symbol, instant_metric.label())
      metric_deque: Deque[(datetime.datetime, List[float])] = self.snapshot_instant_metrics[metric_name]
      metric_deque.append((row.timestamp, values))

      self._flush_output(['snapshot-instant-metric', *metric_name], row.timestamp, values)
    self.__remove_old_metric('snapshot-instant-metric', row.timestamp)

  def _flush_last(self):
    """

    :return: flush contents of metric storage values when dataset is finished
    """

    for name, value in self.snapshot_instant_metrics.items():
      while len(value) > 0:
        self._flush_output(['snapshot-instant-metric', name], *value.popleft())

    for symbol_action_window, metric in self.trade_time_metrics.items():
      while len(metric) > 0:
        # symbol, action, window-size
        self._flush_output(['trade-time-metric', *symbol_action_window], *metric.popleft())

  # todo: that would be great to be tracked by other thread, not main one
  def __remove_old_metric(self, label: str, timestamp: datetime.datetime):
    if label == 'trade-time-metric':
      collection = self.trade_time_metrics
    elif label == 'snapshot-instant-metric':
      collection = self.snapshot_instant_metrics
    elif label == 'snapshot':
      collection = self.memory
    elif label == 'trade':
      collection = self.trades


    for name, deque in collection.items():
      while True:
        if (timestamp - deque[0][0]).seconds >= self.time_horizon:
          # self._flush_output([label, *name], *deque.popleft())
          deque.popleft()
        else:
          break

  def __str__(self):
    return '<Backtest with reader={}>'.format(self.reader)
