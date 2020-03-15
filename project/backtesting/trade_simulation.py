import datetime
from dataclasses import dataclass
from typing import List, Deque, Dict, Tuple, Optional, Union

from utils.data import OrderBook, Trade
from metrics.metrics import InstantMetric, MetricData, TimeMetric
from metrics.filters import Filters

@dataclass
class Order:
  price: float
  volume: int
  side: str

class Strategy:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, instant_metrics: List[InstantMetric],
               filters: List[Filters.Filter] = (Filters.DepthFilter(3), ), time_metrics: Optional[List[TimeMetric]] = None, delay = 400e-6):
    """

    :param instant_metrics:
    :param filters:
    :param delay:
    """
    self.instant_metrics: List[InstantMetric] = instant_metrics
    self.filters: List[Filters.Filter] = filters
    self.time_metrics: List[TimeMetric] = time_metrics if time_metrics is not None else []
    self._delay: int = delay

  def trigger(self, row: OrderBook,
              memory: Dict[Tuple[str], Deque[Tuple[datetime.datetime, Union[OrderBook, Trade]]]],
              # (symbol) -> (timestamp, instant-metric-values)
              snapshot_instant_metrics: Dict[Tuple[str], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action, window-size) -> (timestamp, time-metric-values)
              trade_time_metrics: Dict[Tuple[str, str, int], Deque[Tuple[datetime.datetime, List[float]]]],
              # (symbol, action) -> (timestamp, Trade)
              trades: Dict[Tuple[str, str], Deque[Tuple[datetime.datetime, Trade]]]) -> List: # todo: define type later on
    pass


class StrategyWithTrades(Strategy):
  pass  # todo: consider placed trade and trades arrival

class StrategyWithSnapshot(Strategy):
  pass  # todo: consider only snapshots deltas