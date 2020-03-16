from backtesting.output import Output
from backtesting.readers import Reader
from backtesting.trade_simulation import Strategy, OrderRequest, OrderStatus
from utils.data import OrderBook, Trade
from utils.logger import setup_logger

import datetime
from typing import Dict, List, Deque, Optional, Tuple, Union
from collections import defaultdict, deque, OrderedDict
import random
import numpy as np


logger = setup_logger('<backtest>', 'INFO')


class Backtest:

  def __init__(self, reader: Reader,
               simulation: Strategy,
               output: Optional[Output] = None,
               order_position_policy: str = 'top', # 'random' or 'bottom'
               time_horizon:int=120,
               seed=1337,
               simulation_percentage = 0.15):
    """

    :param reader:
    :param simulation:
    :param output:
    :param order_position_policy:
    :param time_horizon:
    :param seed:
    :param simulation_percentage: how much trade's volume is consumed by simulated order
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

    # (symbol, side) -> price -> List[(order_id, volume-left, consumption-ratio)]
    self.simulated_orders: Dict[Tuple[str, str], OrderedDict[float, List[Tuple[int, float, float]]]] = defaultdict(lambda: defaultdict(list))
    # id -> request
    self.simulated_orders_id: Dict[int, OrderRequest] = {}

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
    self.simulation_percentage = simulation_percentage
    self.__initialize_time_metrics()

    logger.info(f"Initialized {self}")

  def _process_event(self, event: Union[Trade, OrderBook]):
    def filter_snapshot(row: OrderBook) -> bool:
      filtered = True
      for filter in self.simulation.filters:
        if not filter.filter(row):
          filtered = False
          break

      return filtered

    actions = None
    if type(event) is OrderBook:
      if not filter_snapshot(event):
        return
      self._update_memory(event)
      self._update_metrics(event)
      actions = self.simulation.trigger_snapshot(event,
                                                 self.memory, self.snapshot_instant_metrics,
                                                 self.trade_time_metrics, self.trades)
    elif type(event) is Trade:  # it must be cumulative trade if any
      self._update_trades(event)
      statuses = self._evaluate_statuses(event)
      actions = self.simulation.trigger_trade(event, statuses,
                                              self.memory, self.snapshot_instant_metrics,
                                              self.trade_time_metrics, self.trades)

    if actions is not None:
      self._process_actions(actions)


  def run(self):
    logger.info(f'Backtest initialize run')

    for row in self.reader:
      self._process_event(row)

    # self._flush_last()
    logger.info(f'Backtest finished run')

  def _evaluate_statuses(self, trade: Trade) -> List[OrderStatus]:
    """
    Trade simulation unit
    :param trade:
    :return:
    """
    statuses = []

    order_side = 'bid' if trade.side == 'Sell' else 'ask'
    # todo: what about aggressive orders?
    orders = self.simulated_orders[(trade.symbol, order_side)]

    if len(orders) > 0:
      # order_id, volume - left, consumption - ratio
      sorted_orders: List[float, Tuple[int, float, float]] = list(sorted(orders.items(), key=lambda x: x[0]))

      for price, order_requests in sorted_orders:
        for idx, (order_id, volume_left_old, consumption) in enumerate(order_requests):
          order: OrderRequest = self.simulated_orders_id[order_id]
          if (trade.side == 'Sell' and order.side == 'bid' and order.price >= trade.price) or \
              (trade.side == 'Buy' and order.side == 'ask' and order.price <= trade.price):

            volume_left = max(0, volume_left_old - trade.volume)
            if volume_left != 0:
              orders[order.price][idx] = (order_id, volume_left, consumption)
            else:
              consumption += (float(trade.volume) - volume_left_old) / order.volume * self.simulation_percentage   # HERE ARE 15%
              if consumption >= 1.0:  # order is executed
                finished = OrderStatus.finish(order_id, trade.timestamp)
                statuses.append(finished)
                del orders[order.price][idx]
                del self.simulated_orders_id[order.id]
              else:
                orders[order.price][idx] = (order_id, volume_left, consumption)

    return statuses

  def _process_actions(self, actions: List[OrderRequest]):
    for action in actions:
      if action.command == 'new':
        symbol, side, price = action.label()
        orderbook = self.memory[(symbol, )][-1][1] # get most recent (datetime, orderbook) and return orderbook
        if side == 'bid':
          prices = orderbook.bid_prices
          volumes = orderbook.bid_volumes
        elif side == 'ask':
          prices = orderbook.ask_prices
          volumes = orderbook.ask_volumes
        else:
          raise KeyError("wrong side")

        idx = np.where(prices == price)[0]
        level_volume = volumes[idx]

        orders = self.simulated_orders[(symbol, side)][price]
        orders_volume = sum(map(lambda x: x[0].volume * (1.0 - x[1]), map(lambda x: (self.simulated_orders_id[x[0]], x[2]), orders)))
        orders.append((action.id, self._generate_initial_position() * level_volume + orders_volume, 0.0))
        self.simulated_orders_id[action.id] = action
      elif action.command == 'delete':
        order = self.simulated_orders_id.pop(action.id)
        symbol, side, price = order.label()
        self.simulated_orders[(symbol, side)][price].remove(order)

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
