import datetime
from collections import defaultdict
from typing import List, Deque, Dict, Tuple, Optional, Union

from backtesting.data import OrderStatus, OrderRequest
from utils.data import OrderBook, Trade
from metrics.metrics import InstantMetric, TradeMetric, TimeMetric, DeltaMetric, CompositeMetric
from metrics.filters import Filters


class Strategy:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, instant_metrics: List[InstantMetric],
               depth_filter: Filters.DepthFilter = (Filters.DepthFilter(3),),
               time_metrics_trade: List[TradeMetric] = [],
               time_metrics_snapshot: List[DeltaMetric] = [],
               composite_metrics: List[CompositeMetric] = [],
               initial_balance: int = int(1e6),
               delay = 400e-6):
    """

    :param instant_metrics:
    :param depth_filter:
    :param delay:
    """
    self.instant_metrics: List[InstantMetric] = instant_metrics
    self.filter: Filters.DepthFilter = depth_filter
    self.time_metrics: Dict[str, List[TimeMetric]] = {'trade': time_metrics_trade,'orderbook': time_metrics_snapshot}
    self.composite_metrics: List[CompositeMetric] = composite_metrics
    self._delay: int = delay
    self.pending_orders: Dict[int, OrderRequest] = {}
    self.balance: Dict[str, int] = defaultdict(lambda: 0)
    self.balance['USD'] = initial_balance

    self._bind_metrics()

  def _bind_metrics(self):
    metrics_map = {}
    for item in self.instant_metrics:
      metrics_map[item.name] = item
    for item in self.time_metrics['trade']:
      metrics_map[item.name] = item
    for item in self.time_metrics['orderbook']:
      metrics_map[item.name] = item

    for item in self.composite_metrics:
      metrics_map[item.name] = item

    for item in self.composite_metrics:
      item.set_metric_map(metrics_map)

  def _update_balance_statuses(self, statuses: List[OrderStatus]):
    for status in statuses:
      if status.status == 'finished':
        order = self.pending_orders[status.id]
        if order.side == 'ask':
          self.balance['USD'] += order.volume
        elif order.side == 'bid':
          self.balance[order.symbol] += order.volume / order.price

  def _update_balance_orders(self, orders: List[OrderRequest]):
    for order in orders:
      self.pending_orders[order.id] = order
      if order.side == 'ask':
        self.balance[order.symbol] -= order.volume / order.price
      elif order.side == 'bid':
        self.balance['USD'] -= order.volume

  def define_orders(self, row: Union[Trade, OrderBook], # todo: add delay
                    memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
                    snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
                    trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
                    trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]) -> List[OrderRequest]:
    return []

  def trigger_trade(self, row: Trade, # todo: add delay
                    statuses: List[OrderStatus],
                    memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
                    # (symbol) -> (timestamp, instant-metric-values)
                    snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
                    # (symbol, action, window-size) -> (timestamp, time-metric-values)
                    trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
                    # (symbol, action) -> (timestamp, Trade)
                    trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]) -> List[OrderRequest]:
    self._update_balance_statuses(statuses)
    orders = self.define_orders(row, memory, snapshot_instant_metrics, trade_time_metrics, trades)
    self._update_balance_orders(orders)
    return orders

  def trigger_snapshot(self, row: OrderBook,
              memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
              # (symbol) -> (timestamp, instant-metric-values)
              snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action, window-size) -> (timestamp, time-metric-values)
              trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action) -> (timestamp, Trade)
              trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]) -> List[OrderRequest]:
    orders = self.define_orders(row, memory, snapshot_instant_metrics, trade_time_metrics, trades)
    self._update_balance_orders(orders)
    return orders
