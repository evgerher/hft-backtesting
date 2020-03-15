import datetime
from dataclasses import dataclass
from typing import List, Deque, Dict, Tuple, Optional, Union

from utils.data import OrderBook, Trade
from metrics.metrics import InstantMetric, MetricData, TimeMetric
from metrics.filters import Filters

initial_id = 0

@dataclass
class OrderStatus:
  id: int
  status: str
  at: datetime.datetime

  @staticmethod
  def finish(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, 'finished', timestamp)

  @staticmethod
  def remove(id: int, timestamp: datetime.datetime) -> 'OrderStatus':
    return OrderStatus(id, 'removed', timestamp)

@dataclass
class OrderRequest:
  id: int
  command: str
  price: float
  volume: int
  symbol: str
  side: str
  created: datetime.datetime

  def label(self) -> Tuple[str, str, float]:
    return (self.symbol, self.side, self.price)

  @staticmethod
  def _generate_id() -> int:
    global initial_id
    id = initial_id
    initial_id += 1
    return id

  @staticmethod
  def cancelOrder(id: int) -> 'OrderRequest':
    return OrderRequest(id, 'delete', None, None, None, None, None)

  @staticmethod
  def create(price: float, volume: int, symbol: str, side: str, timestamp: datetime.datetime) -> 'OrderRequest':
    id = OrderRequest._generate_id()
    return OrderRequest(id, 'new', price, volume, symbol, side, timestamp)

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
    self.balance: int = initial_balance

  def trigger_trade(self, row: Trade,
              statuses: List[OrderStatus],
              memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
              # (symbol) -> (timestamp, instant-metric-values)
              snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action, window-size) -> (timestamp, time-metric-values)
              trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action) -> (timestamp, Trade)
              trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]) -> List[OrderRequest]:
    pass

  def trigger_snapshot(self, row: Trade,
              memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
              # (symbol) -> (timestamp, instant-metric-values)
              snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action, window-size) -> (timestamp, time-metric-values)
              trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action) -> (timestamp, Trade)
              trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]) -> List[OrderRequest]:
    pass
