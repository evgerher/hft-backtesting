import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Callable, Optional
import copy

from hft.backtesting.data import OrderStatus, OrderRequest
from hft.units.metrics.composite import CompositeMetric
from hft.units.metrics.instant import DeltaMetric, InstantMetric
from hft.units.metrics.time import TradeMetric, TimeMetric
from hft.utils.consts import Statuses, QuoteSides
from hft.utils.data import OrderBook, Trade
from hft.units.filters import Filters
from hft.utils.logger import setup_logger

from abc import ABC, abstractmethod
logger = setup_logger('<Strategy>')

@dataclass
class PerpetualFee:
  __slots__ = ['maker', 'taker', 'long', 'short', '__weakref__']
  maker: float
  taker: float
  long: float
  short: float

  @staticmethod
  def Bitmex_XBT():
    return PerpetualFee(-0.025 / 100, 0.075 / 100, -0.0385 / 100, 0.0385 / 100)

  @staticmethod
  def Bitmex_ETH():
    return PerpetualFee(-0.025 / 100, 0.075 / 100, 0.01 / 100, -0.01 / 100)

  @staticmethod
  def zero():
    return PerpetualFee(0, 0, 0, 0)

@dataclass
class TraditionalFee:
  __slots__ = ['maker', 'taker', 'settlement', '__weakref__']
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

  def __init__(self, instant_metrics: List[InstantMetric] = [],
               delta_metrics: List[DeltaMetric] = [],
               time_metrics_trade: List[TradeMetric] = [],
               time_metrics_snapshot: List[TimeMetric] = [],
               composite_metrics: List[CompositeMetric] = [],
               initial_balance: float = 1e6,
               balance_listener: Callable[[Tuple], None] = None,
               filter: Optional[Union[int, Filters.Filter]] = 4):
    """

    :param instant_metrics:
    :param depth_filter:
    """
    if isinstance(filter, int):
      self.filter = Filters.DepthFilter(filter)
    elif isinstance(filter, Filters.Filter):
      self.filter = filter
    else:
      self.filter = Filters.DepthFilter(4)

    self.instant_metrics: List[InstantMetric] = instant_metrics
    self.time_metrics: Dict[str, List[TimeMetric]] = {
      'trade': time_metrics_trade,
      'orderbook': time_metrics_snapshot
    }
    self.composite_metrics: List[CompositeMetric] = composite_metrics
    self.delta_metrics = delta_metrics

    self.active_orders: Dict[int, OrderRequest] = {}

    self.balance: Dict[str, float] = {'USD': initial_balance, 'XBTUSD': 0.0, 'ETHUSD': 0.0}
    self.position: Dict[str, Tuple[float, float]] = {'XBTUSD': (0.0, 0.0), 'ETHUSD': (0.0, 0.0)}  # (average_price, volume)

    self.balance_listener = balance_listener
    self.fee: Dict[str, PerpetualFee] = defaultdict(lambda: PerpetualFee.zero())
    self.fee['XBTUSD'] = PerpetualFee.Bitmex_XBT()
    self.fee['ETHUSD'] = PerpetualFee.Bitmex_ETH()
    self.pennies = 0.0

    self.metrics_map = self.__bind_metrics()

  def __bind_metrics(self): # todo: refactor binding into one line
    metrics_map = {}
    for item in self.instant_metrics:
      metrics_map[item.name] = item

    for sublist in self.time_metrics.values():
      for item in sublist:
        metrics_map[item.name] = item

    for m in self.delta_metrics:
      metrics_map[m.name] = m

    for item in self.composite_metrics:
      metrics_map[item.name] = item

    for item in self.composite_metrics:
      item.set_metric_map(metrics_map)

    return metrics_map

  def _balance_update_by_status(self, statuses: List[OrderStatus]):
    # direction_mappings = {
    #   ('ask', 'partial') : +'USD', -symbol,
    #   ('bid', 'partial') : +order.symbol, -USD,
    #   ('ask', 'finished'): +'USD', -symbol,
    #   ('bid', 'finished'): +order.symbol, -USD,
    #   ('ask', 'cancel')  : +order.symbol, -USD
    #   ('bid', 'cancel')  : +'USD', -symbol
    # }

    for status in statuses:
      # if status.status != Statuses.CANCEL: # finished and partial
      logger.debug(f'Received status: {status}')
      order = self.active_orders.get(status.id, None)

      if order is not None:
        if status.status != Statuses.PARTIAL: # finished and cancel
          volume = order.volume - order.volume_filled
        else: # reminder - here are only partial
          volume = status.volume
          order.volume_filled += status.volume

        converted_volume = volume / order.price
        if status.status == Statuses.CANCEL:
          converted_volume  = -converted_volume
          volume            = -volume
        else:
          maker_fee = volume * self.fee[order.symbol].maker  # negative
          self.balance['USD'] -= maker_fee
          self.pennies -= maker_fee

        avg_price, vol = self.position[order.symbol]

        if order.side == QuoteSides.ASK:
          self.balance['USD'] += volume
          self.balance[order.symbol] -= converted_volume
          # Contracts * Multiplier * (1/Entry Price - 1/Exit Price)

          if status.status != Statuses.CANCEL:
            total_volume = vol - converted_volume
            if vol <= 0:
              ratio = converted_volume / total_volume
              self.position[order.symbol] = ((avg_price * (1.0 - ratio) + order.price * ratio), total_volume)
            else:
              if total_volume <= 0:
                self.position[order.symbol] = (order.price, total_volume)
              else:
                self.position[order.symbol] = (avg_price, total_volume)


        else: # bid
          self.balance[order.symbol] += converted_volume
          self.balance['USD'] -= volume

          if status.status != Statuses.CANCEL:
            total_volume = vol + converted_volume
            if vol >= 0:
              ratio = converted_volume / total_volume
              self.position[order.symbol] = ((avg_price * (1.0 - ratio) + order.price * ratio), total_volume)
            else:
              if total_volume >= 0:
                self.position[order.symbol] = (order.price, total_volume)
              else:
                self.position[order.symbol] = (avg_price, total_volume)

  def _balance_update_new_order(self, orders: List[OrderRequest]):
    for order in orders:
      if order.command != Statuses.CANCEL:
        logger.info(f'New order: {order}')
        self.active_orders[order.id] = order

  def _get_allowed_volume(self, symbol, memory, side):
    latest: OrderBook = memory[('orderbook', symbol)]
    side_level = latest.bid_volumes[0] if side == QuoteSides.BID else latest.ask_volumes[0]
    return int(max(side_level * 0.15, 10000))

  def __validate_orders(self, orders, memory):
    for order in orders:
      if order.command != Statuses.CANCEL:
        latest: OrderBook = memory[('orderbook', order.symbol)]
        side_level = latest.bid_volumes[0] if order.side == QuoteSides.BID else latest.ask_volumes[0]
        assert order.volume > 0 and ((side_level * 0.15 >= order.volume) or order.volume <= 10000), \
          f"order size must be max 15% of the level or 10000 units: {order.volume}, {side_level}"

  def __remove_finished_orders(self, statuses):
    for status in statuses:
      if status.status == Statuses.FINISHED or status.status == Statuses.CANCEL:
        try:
          del self.active_orders[status.id]
        except:
          pass

  @abstractmethod
  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]],
                    is_trade: bool) -> List[OrderRequest]:
    raise NotImplementedError

  def trigger(self, row: Union[Trade, OrderBook],
               statuses: List[OrderStatus],
               memory: Dict[str, Union[Trade, OrderBook]],
               is_trade: bool):
    self._balance_update_by_status(statuses)
    orders = self.define_orders(row, statuses, memory, is_trade)
    self.__validate_orders(orders, memory)
    self.__remove_finished_orders(statuses)
    self._balance_update_new_order(orders)
    # balance updated, notify listener
    self._balance_listener(memory, row.timestamp, orders, statuses)

    orders = [copy.deepcopy(t) for t in orders]
    return orders

  @abstractmethod
  def _balance_listener(self, memory: Dict[str, Union[Trade, OrderBook]],
                        ts: datetime.datetime,
                        orders: List[OrderRequest],
                        statuses: List[OrderStatus]):
    raise NotImplementedError

  def return_unfinished(self, statuses: List[OrderStatus], memory: Dict[str, Union[Trade, OrderBook]]):
    logger.info('Update balance with unfinished tasks')
    self._balance_update_by_status(statuses)

    self._balance_listener(memory, memory['orderbook', 'XBTUSD'].timestamp, [], statuses)


class CalmStrategy(Strategy):
  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]],
                    is_trade: bool):
    return []

  def trigger(self, *args):
    return []

class CancelsApplied:
  def __init__(self, strategy: Strategy, cancels: List[OrderRequest]):
    # assert all([t.command == Statuses.CANCEL for t in cancels])
    self.strategy: Strategy = strategy
    orders = [strategy.active_orders[x.id] for x in cancels]
    self.cancel_statuses = [OrderStatus.partial(x.id, x.created, y.volume_filled, y.volume - y.volume_filled) for x, y in zip(cancels, orders)]
    self.revert_statuses = [OrderStatus.partial(x.id, x.created, y.volume_filled, y.volume_filled - y.volume) for x, y in zip(cancels, orders)]

  def __enter__(self):
    self.strategy._balance_update_by_status(self.cancel_statuses)

  def __exit__(self, type, value, traceback):
    self.strategy._balance_update_by_status(self.revert_statuses)
