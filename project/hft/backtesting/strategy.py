from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Callable

from hft.backtesting.data import OrderStatus, OrderRequest
from hft.utils.consts import Statuses, QuoteSides
from hft.utils.data import OrderBook, Trade
from hft.metrics.metrics import InstantMetric, TradeMetric, TimeMetric, DeltaMetric, CompositeMetric
from hft.metrics.filters import Filters
from hft.utils.logger import setup_logger

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
               balance_listener: Callable[[Tuple], None] = None,
               filter_depth: int = None):
    """

    :param instant_metrics:
    :param depth_filter:
    """
    self.instant_metrics: List[InstantMetric] = instant_metrics
    if filter_depth is not None:
      self.filter = Filters.DepthFilter(filter_depth)
    else:
      self.filter: Filters.DepthFilter = depth_filter
    self.time_metrics: Dict[str, List[TimeMetric]] = {
      'trade': time_metrics_trade,
      'orderbook': time_metrics_snapshot
    }
    self.composite_metrics: List[CompositeMetric] = composite_metrics

    self.active_orders: Dict[int, OrderRequest] = {}

    self.balance: Dict[str, float] = defaultdict(lambda: 0)
    self.balance['USD'] = initial_balance
    self.position: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

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
      if status.status != Statuses.CANCEL: # finished and partial
        logger.debug(f'Received status: {status}')
        order = self.active_orders[status.id]

        if status.status != Statuses.PARTIAL: # finished and cancel
          volume = order.volume - order.volume_filled
        else: # todo: reminder - partial else always
          volume = status.volume_total
          order.volume_filled += status.volume

        converted_volume = volume / order.price
        avg_price, vol = self.position[order.symbol]

        if order.side == QuoteSides.ASK:
          self.balance['USD'] += volume
          self.balance[order.symbol] -= converted_volume
          # Contracts * Multiplier * (1/Entry Price - 1/Exit Price)

          if vol <= 0:
            total_volume = vol - converted_volume
            ratio = converted_volume / total_volume
            self.position[order.symbol] = ((avg_price * (1.0 - ratio) + order.price * ratio), total_volume)
          else:
            total_volume = vol - converted_volume
            if total_volume <= 0:
              self.position[order.symbol] = (order.price, total_volume)
            else:
              self.position[order.symbol] = (avg_price, total_volume)


        else: # bid
          self.balance[order.symbol] += converted_volume
          self.balance['USD'] -= volume

          if vol >= 0:
            total_volume = vol + converted_volume
            ratio = converted_volume / total_volume
            self.position[order.symbol] = ((avg_price * (1.0 - ratio) + order.price * ratio), total_volume)
          else:
            total_volume = vol + converted_volume
            if total_volume >= 0:
              self.position[order.symbol] = (order.price, total_volume)
            else:
              self.position[order.symbol] = (avg_price, total_volume)

  def _balance_update_new_order(self, orders: List[OrderRequest]):
    for order in orders:
      logger.info(f'New order: {order}')
      ### action negative update balance
      self.active_orders[order.id] = order
      if order.side == QuoteSides.ASK:
        self.balance[order.symbol] += order.volume / order.price * (self.fee[order.symbol].settlement  + self.fee[order.symbol].maker)
      elif order.side == QuoteSides.BID:
        self.balance['USD'] += order.volume * (self.fee[order.symbol].settlement + self.fee[order.symbol].maker)

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
                    memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
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
      # balance = memory[('orderbook', 'XBTUSD')].bid_prices[0] * self.balance['XBTUSD'] + \
      #   memory[('orderbook', 'ETHUSD')].bid_prices[0] * self.balance['ETHUSD'] + \
      #   self.balance['USD']

      midpoint_eth = (memory[('orderbook', 'ETHUSD')].bid_prices[0] + memory[('orderbook', 'ETHUSD')].ask_prices[0]) / 2
      midpoint_xbt = (memory[('orderbook', 'XBTUSD')].bid_prices[0] + memory[('orderbook', 'XBTUSD')].ask_prices[0]) / 2

      state = (self.balance['USD'], self.balance['XBTUSD'], self.balance['ETHUSD'], *self.position['XBTUSD'], *self.position['ETHUSD'], midpoint_xbt, midpoint_eth, row.timestamp)
      # self.balance_listener(self.balance['USD'], row.timestamp)
      self.balance_listener(state)
    return orders

  def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
    logger.info('Update balance with unfinished tasks')
    self._balance_update_by_status(statuses)
    if self.balance_listener is not None and len(statuses) > 0:
      midpoint_eth = (memory[('orderbook', 'ETHUSD')].bid_prices[0] + memory[('orderbook', 'ETHUSD')].ask_prices[0]) / 2
      midpoint_xbt = (memory[('orderbook', 'XBTUSD')].bid_prices[0] + memory[('orderbook', 'XBTUSD')].ask_prices[0]) / 2

      state = (self.balance['USD'], self.balance['XBTUSD'], self.balance['ETHUSD'], *self.position['XBTUSD'], *self.position['ETHUSD'], midpoint_xbt, midpoint_eth, statuses[0].at)
      # self.balance_listener(self.balance['USD'], row.timestamp)
      self.balance_listener(state)


class CalmStrategy(Strategy):
  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]]):
    return []

  def trigger(self, *args):
    return []
