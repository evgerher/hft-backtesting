import datetime
from typing import List, Deque, Dict, Tuple, Optional, Union

from backtesting.data import OrderStatus, OrderRequest
from utils.data import OrderBook, Trade
from metrics.metrics import InstantMetric, TimeMetric
from metrics.filters import Filters


class Strategy:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, instant_metrics: List[InstantMetric],
               filters: List[Filters.Filter] = (Filters.DepthFilter(3), ),
               time_metrics: Optional[List[TimeMetric]] = None,
               initial_balance: int = int(1e6),
               delay = 400e-6):
    """

    :param instant_metrics:
    :param filters:
    :param delay:
    """
    self.instant_metrics: List[InstantMetric] = instant_metrics
    self.filters: List[Filters.Filter] = filters
    self.time_metrics: List[TimeMetric] = time_metrics if time_metrics is not None else []
    self._delay: int = delay
    self.pending_orders: Dict[int, OrderRequest] = {}
    self.balance: Dict[str, int] = {'USD': initial_balance, 'XBT': 0}

  def _update_balance_statuses(self, statuses: List[OrderStatus]):
    for status in statuses:
      if status.status == 'finished':
        order = self.pending_orders[status.id]
        if order.side == 'Sell':
          self.balance['USD'] += order.volume
          # self.balance[order.symbol] -= order.volume / order.price
        elif order.side == 'Buy':
          # self.balance[order.symbol] -= order.volume
          self.balance[order.symbol] += order.volume / order.price

  def _update_balance_orders(self, orders: List[OrderRequest]):
    for order in orders:
      if order.side == 'Sell':
        self.balance[order.symbol] -= order.volume / order.price
      elif order.side == 'Buy':
        self.balance[order.symbol] -= order.volume

  def define_orders(self, *args) -> List[OrderRequest]:
    pass

  def trigger_trade(self, row: Trade,
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

  def trigger_snapshot(self, row: Trade,
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