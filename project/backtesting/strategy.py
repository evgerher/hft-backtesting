import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Callable

from backtesting.data import OrderStatus, OrderRequest
from utils.consts import Statuses, QuoteSides
from utils.data import OrderBook, Trade
from metrics.metrics import InstantMetric, TradeMetric, TimeMetric, DeltaMetric, CompositeMetric
from metrics.filters import Filters
from utils.logger import setup_logger

from abc import ABC, abstractmethod
logger = setup_logger('<Strategy>')

@dataclass
class PerpetualFee:
  maker: float
  taker: float
  long: float
  short: float

  @staticmethod
  def Bitmex_XBT():
    return PerpetualFee(-0.025, 0.075, -0.0385, 0.0385)

  @staticmethod
  def Bitmex_ETH():
    return PerpetualFee(-0.025, 0.075, 0.01, -0.01)

  @staticmethod
  def zero():
    return PerpetualFee(0, 0, 0, 0)

@dataclass
class TraditionalFee:
  maker: float
  taker: float
  settlement: float

  @staticmethod
  def Bitmex_XBT():
    return TraditionalFee(-0.025 / 100, 0.075 / 100, 0.05 / 100)

  @staticmethod
  def Bitmex_ETH():
    return TraditionalFee(-0.05 / 100, 0.25 / 100, 0)

  @staticmethod
  def zero():
    return TraditionalFee(0, 0, 0)


class Strategy(ABC):
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def __init__(self, instant_metrics: List[InstantMetric] = [],
               depth_filter: Filters.DepthFilter = Filters.DepthFilter(4),
               time_metrics_trade: List[TradeMetric] = [],
               time_metrics_snapshot: List[DeltaMetric] = [],
               composite_metrics: List[CompositeMetric] = [],
               initial_balance: int = int(1e6),
               balance_listener: Callable[[float, datetime.datetime], None] = None):
    """

    :param instant_metrics:
    :param depth_filter:
    """
    self.instant_metrics: List[InstantMetric] = instant_metrics
    self.filter: Filters.DepthFilter = depth_filter
    self.time_metrics: Dict[str, List[TimeMetric]] = {
      'trade': time_metrics_trade,
      'orderbook': time_metrics_snapshot
    }
    self.composite_metrics: List[CompositeMetric] = composite_metrics

    self.active_orders: Dict[int, OrderRequest] = {}

    self.balance: Dict[str, int] = defaultdict(lambda: 0)
    self.balance['USD'] = initial_balance
    self.balance_listener = balance_listener
    self.fee: Dict[str, TraditionalFee] = defaultdict(lambda: TraditionalFee.zero())
    self.fee['XBTUSD'] = TraditionalFee.Bitmex_XBT()
    self.fee['ETHUSD'] = TraditionalFee.Bitmex_ETH()

    self.metrics_map = self.__bind_metrics()

  def __bind_metrics(self):
    metrics_map = {}
    for item in self.instant_metrics:
      metrics_map[item.name] = item

    for sublist in self.time_metrics.values():
      for item in sublist:
        metrics_map[item.name] = item

    for item in self.composite_metrics:
      metrics_map[item.name] = item

    for item in self.composite_metrics:
      item.set_metric_map(metrics_map)

    return metrics_map

  def _balance_update_by_status(self, statuses: List[OrderStatus]):
    # direction_mappings = {
    #   ('ask', 'partial') : 'USD',
    #   ('bid', 'partial') : order.symbol,
    #   ('ask', 'finished'): 'USD',
    #   ('bid', 'finished'): order.symbol,
    #   ('ask', 'cancel')  : order.symbol,
    #   ('bid', 'cancel')  : 'USD',
    # }

    for status in statuses:
      # logger.debug(f'Received status: {status}')
      order = self.active_orders[status.id]
      if status.status != Statuses.PARTIAL: # finished and cancel
        volume = order.volume - order.volume_filled
      else: # todo: reminder - partial else always
        volume = status.volume_total
        order.volume_filled += status.volume

      ### action positive update balance

      if status.status == Statuses.CANCEL:
        logger.info(f'Cancel order: {status}')
        if order.side == QuoteSides.BID:
          self.balance['USD'] += volume
        else: # ask
          self.balance[order.symbol] += volume / order.price
      else:
        if order.side == QuoteSides.ASK:
          self.balance['USD'] += volume
        elif order.side: # bid
          self.balance[order.symbol] += volume / order.price

  def _balance_update_new_order(self, orders: Tuple[OrderRequest]):
    for order in orders:
      logger.info(f'New order: {order}')
      ### action negative update balance
      self.active_orders[order.id] = order
      if order.side == QuoteSides.ASK:
        self.balance[order.symbol] -= order.volume / order.price + order.volume * (self.fee[order.symbol].settlement  + self.fee[order.symbol].maker)
      elif order.side == QuoteSides.BID:
        self.balance['USD'] -= order.volume + order.volume * (self.fee[order.symbol].settlement + self.fee[order.symbol].maker)

  def _get_allowed_volume(self, symbol, memory, side):
    latest: OrderBook = memory[('orderbook', symbol)]
    side_level = latest.bid_volumes[0] if side == QuoteSides.BID else latest.ask_volumes[0]
    return int(max(side_level * 0.15, 10000))

  def __validate_orders(self, orders, memory):
    for order in orders:
      latest: OrderBook = memory[('orderbook', order.symbol)]
      side_level = latest.bid_volumes[0] if order.side == QuoteSides.BID else latest.ask_volumes[0]
      assert order.volume > 0 and ((side_level * 0.15 >= order.volume) or order.volume <= 10000), \
        f"order size must be max 15% of the level or 10000 units: {order.volume}, {side_level}"

  def __remove_finished_orders(self, statuses):
    for status in statuses:
      if status.status == Statuses.FINISHED or status.status == Statuses.CANCEL:
        del self.active_orders[status.id]

  @abstractmethod
  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]]) -> Tuple[OrderRequest]:
    raise NotImplementedError

  def trigger(self, row: Union[Trade, OrderBook],
               statuses: List[OrderStatus],
               memory: Dict[str, Union[Trade, OrderBook]]):
    self._balance_update_by_status(statuses)
    orders = self.define_orders(row, statuses, memory)
    self.__validate_orders(orders, memory)
    self._balance_update_new_order(orders)
    self.__remove_finished_orders(statuses)

    # balance updated, notify listener
    if self.balance_listener is not None and (len(orders) > 0 or len(statuses) > 0) and len(memory) >= 4:
      balance = memory[('orderbook', 'XBTUSD')].bid_prices[0] * self.balance['XBTUSD'] + \
        memory[('orderbook', 'ETHUSD')].bid_prices[0] * self.balance['ETHUSD'] + \
        self.balance['USD']
      # self.balance_listener(self.balance['USD'], row.timestamp)
      self.balance_listener(balance, row.timestamp)
    return orders

  def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
    logger.info('Update balance with unfinished tasks')
    self._balance_update_by_status(statuses)
    if self.balance_listener is not None:
      balance = memory[('orderbook', 'XBTUSD')].bid_prices[0] * self.balance['XBTUSD'] + \
                memory[('orderbook', 'ETHUSD')].bid_prices[0] * self.balance['ETHUSD'] + \
                self.balance['USD']
      # self.balance_listener(self.balance['USD'], statuses[0].at)
      self.balance_listener(balance, statuses[0].at)


class CalmStrategy(Strategy):
  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]]):
    return []

  def trigger(self, *args):
    return []
