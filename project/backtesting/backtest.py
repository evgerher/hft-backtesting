from backtesting.output import Output
from backtesting.readers import Reader
from backtesting.strategy import Strategy
from backtesting.data import OrderStatus, OrderRequest
from utils.types import Delta
from utils.data import OrderBook, Trade
from utils.logger import setup_logger

import datetime
from typing import Dict, List, Optional, Tuple, Union, Deque
from collections import defaultdict, OrderedDict, deque
import random
import numpy as np

from utils.types import SymbolSide, OrderState

logger = setup_logger('<backtest>', 'INFO')


class Backtest:

  def __init__(self, reader: Reader,
               simulation: Strategy,
               output: Optional[Output] = None,
               order_position_policy: str = 'top', # 'random' or 'bottom'
               time_horizon:int=120,
               seed=1337,
               notify_partial = True,
               delay=0):
    """

    :param reader:
    :param simulation:
    :param output:
    :param order_position_policy:
    :param time_horizon:
    :param seed:
    :param delay:
    """
    self.reader: Reader = reader
    self.simulation: Strategy = simulation
    self.time_horizon: int = time_horizon

    self.memory: Dict[str, Union[Trade, OrderBook]] = {}
    self.output: Output = output

    self.pending_orders: Deque[(datetime.datetime, OrderRequest)] = deque()
    self.pending_statuses: Deque[(datetime.datetime, OrderStatus)] = deque()
    # (symbol, side) -> price -> List[(order_id, volume-left, consumption-ratio)]
    self.simulated_orders: Dict[SymbolSide, OrderedDict[float, List[OrderState]]] = defaultdict(lambda: defaultdict(list))
    # id -> request
    self.simulated_orders_id: Dict[int, OrderRequest] = {}
    self._notify_partial = notify_partial

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
    self.delay = delay
    logger.info(f"Initialized {self}")

  def _process_event(self, event: Union[Trade, OrderBook]):
    actions = []
    option = None
    if isinstance(event, OrderBook):
      option = self.simulation.filter.process(event)
      if option is None:
        return

      if self.delay != 0: # todo: may be remove it? It will make slight performance lowerance
        # update pending orders, if delay passed
        pend_orders = self.__update_pending_objects(event.timestamp, self.pending_orders)
        for ord in pend_orders:
          self.__move_order_to_active(ord)
      self._update_snapshots(event, option)
      actions = self.simulation.trigger_snapshot(event, self.memory)
    elif isinstance(event, Trade):
      self._update_trades(event)
      statuses = self._evaluate_statuses(event)

      if self.delay != 0: # if delay, statuses are also queued
        self.pending_statuses.append(statuses)
        statuses = self.__update_pending_objects(event.timestamp, self.pending_statuses)

      actions = self.simulation.trigger_trade(event, statuses, self.memory)

    self._update_composite_metrics(event, option)
    if len(actions) > 0:
      self._process_actions(actions)


  def run(self):
    logger.info(f'Backtest initialize run')
    for row in self.reader:
      self._process_event(row)
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
      sorted_orders: List[float, OrderState] = list(sorted(orders.items(), key=lambda x: x[0]))
      remove_finished = []
      for price, order_requests in sorted_orders:
        for idx, (order_id, volume_level_old, consumption) in enumerate(order_requests):
          order: OrderRequest = self.simulated_orders_id[order_id]
          if (trade.side == 'Sell' and order.side == 'bid' and order.price >= trade.price) or \
              (trade.side == 'Buy' and order.side == 'ask' and order.price <= trade.price):

            volume_left = max(0, volume_level_old - trade.volume)
            if volume_left != 0:
              orders[order.price][idx] = (order_id, volume_left, consumption)
            else:
              consumption += (float(trade.volume) - volume_level_old) / order.volume
              if consumption >= 1.0:  # order is executed
                finished = OrderStatus.finish(order_id, trade.timestamp)
                statuses.append(finished)
                remove_finished.append((order.price, idx))
                del self.simulated_orders_id[order.id]
              else:
                orders[order.price][idx] = (order_id, volume_left, consumption)
                if self._notify_partial:
                  partial = OrderStatus.partial(order_id, trade.timestamp, int(consumption * order.volume))
                  statuses.append(partial)

      for price, idx in remove_finished:
        del orders[price][idx]

    return statuses

  def __move_order_to_active(self, action: OrderRequest):
    symbol, side, price = action.label()
    orderbook = self.memory[('orderbook', symbol)]  # get most recent (datetime, orderbook) and return orderbook
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
    orders_volume = sum(map(lambda x: x[0].volume * (1.0 - x[1]),
                            map(lambda x: (self.simulated_orders_id[x[0]], x[2]),
                                orders))) # todo: add explanation
    orders.append((action.id, self._generate_initial_position() * level_volume + orders_volume, 0.0))
    self.simulated_orders_id[action.id] = action

  def __update_pending_objects(self, timestamp: datetime.datetime, objects_deque: Deque):
    t = timestamp - datetime.timedelta(seconds=self.delay)
    objs = []
    while len(objects_deque) > 0 and t > objects_deque[0]:
      objs.append(objects_deque.popleft())
    return objs


  def _process_actions(self, actions: List[OrderRequest]):
    for action in actions:
      if action.command == 'new':
        if self.delay == 0:
          self.__move_order_to_active(action)
        else:
          self.pending_orders.append((action.created, action))
      elif action.command == 'delete':
        order = self.simulated_orders_id.pop(action.id)
        symbol, side, price = order.label()
        self.simulated_orders[(symbol, side)][price].remove(order)

  def __initialize_time_metrics(self):
    for metrics in self.simulation.time_metrics.values():
      for metric in metrics:
        metric.set_starting_moment(self.reader.initial_moment)

  def _flush_output(self, labels: List[str], timestamp: datetime.datetime, values: List[float]):
    """

    :param timestamp:
    :param object: may be Metric/Snapshot/Trade
    :return:
    """
    if self.output is not None:
      self.output.consume(labels, timestamp, values)

  def _update_trades(self, row: Trade):
    logger.debug(f'Update memory with trade symbol={row.symbol}, side={row.side} @ {row.timestamp}')
    self.memory[('trade', row.symbol, row.side)] = row
    self._flush_output(['trade'], row.timestamp, row)  # todo: fix

    for time_metric in self.simulation.time_metrics['trade']:
      values = time_metric.evaluate(row)
      self._flush_output(['time-metric', 'trade', row.symbol, time_metric.name], row.timestamp, values)

  def _update_composite_metrics(self, data: Union[Trade, OrderBook], option: Optional[Delta]):
    logger.debug('Update composite metrics')

    if isinstance(data, OrderBook):
      for composite_metric in self.simulation.composite_metrics:
        value = composite_metric.evaluate(data)
        self._flush_output(['composite-metric', 'snapshot', data.symbol, composite_metric.name], data.timestamp, [value])

  def _update_snapshots(self, row: OrderBook, option: Delta):
    logger.debug(f'Update metrics with snapshot symbol={row.symbol} @ {row.timestamp}')
    self.memory[('orderbook', row.symbol)] = row
    self._flush_output(['snapshot'], row.timestamp, row)  # todo: fix

    for instant_metric in self.simulation.instant_metrics:
      values = instant_metric.evaluate(row)
      self._flush_output(['instant-metric', 'snapshot', row.symbol, instant_metric.name], row.timestamp, values)

    if option[-1] != 0: # if volume altered on best level
      for time_metric in self.simulation.time_metrics['orderbook']:
        values = time_metric.evaluate(option)
        self._flush_output(['delta', 'snapshot', row.symbol, time_metric.name], row.timestamp, values)

  def __str__(self):
    return '<Backtest with reader={}>'.format(self.reader)
