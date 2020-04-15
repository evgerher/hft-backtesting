from hft.backtesting.output import Output
from hft.backtesting.readers import OrderbookReader
from hft.backtesting.strategy import Strategy
from hft.backtesting.data import OrderStatus, OrderRequest
from hft.utils.consts import Statuses, QuoteSides, TradeSides
from hft.utils.types import Delta
from hft.utils.data import OrderBook, Trade
from hft.utils.logger import setup_logger
from tqdm import tqdm

import datetime
from typing import Dict, List, Optional, Union, Deque
from collections import defaultdict, OrderedDict, deque
import random
import numpy as np

from hft.utils.types import SymbolSide, OrderState

logger = setup_logger('<backtest>', 'INFO')


class Backtest:

  def __init__(self, reader: OrderbookReader,
               simulation: Strategy,
               output: Optional[Output] = None,
               order_position_policy: str = 'tail', # 'random' or 'head'
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
    :param delay: delay in microseconds !
    """
    self.reader: OrderbookReader = reader
    self.simulation: Strategy = simulation
    self.time_horizon: int = time_horizon

    self.memory: Dict[str, Union[Trade, OrderBook]] = {}
    self.output: Output = output

    self.pending_orders: Deque[(datetime.datetime, OrderRequest)] = deque()
    self.pending_statuses: Deque[(datetime.datetime, OrderStatus)] = deque()
    # (symbol, side) -> price -> List[(order_id, volume_total-left, consumption-ratio)]
    self.simulated_orders: Dict[SymbolSide, OrderedDict[float, List[OrderState]]] = defaultdict(lambda: defaultdict(list))
    # id -> request
    self.simulated_orders_id: Dict[int, OrderRequest] = {}
    self._notify_partial = notify_partial
    self.price_step: Dict[str, float] = {'XBTUSD': 0.5, 'ETHUSD': 0.05, 'test': 5}
    self.__last_is_trade = defaultdict(lambda: False)

    self.random = random.Random()
    self.random.seed(seed)

    if order_position_policy == 'tail':
      policy = lambda: 1.0
    elif order_position_policy == 'head':
      policy = lambda: 0.0
    elif order_position_policy == 'random':
      policy = lambda: self.random.uniform(0.0, 1.0)
    else:
      policy = lambda: 1.0

    self._generate_initial_position = policy
    self.__initialize_time_metrics()
    self.delay = delay
    logger.info(f"Initialized {self}")

  def _process_event(self, event: Union[Trade, OrderBook], isorderbook: bool):
    statuses = []
    delta = None
    if isorderbook:
      delta = self.simulation.filter.process(event)
      if delta is None:
        return

      if self.delay != 0:
        # update pending orders, if delay passed
        pend_orders = self.__update_pending_objects(event.timestamp, self.pending_orders)
        for ord in pend_orders:
          self.__move_order_to_active(ord)
      if delta[2] >= 4: # is critical
        statuses = self.__critical_price_change(event.symbol, delta[2] % 2, event.timestamp, delta[-1])
      elif delta[2] >= 2: # BID-ALTER or ASK-ALTER
        statuses = self._price_step_cancel_order(event, delta)

      if delta[-1].size > 0 and delta[-1][1, 0] < 0 and not self.__last_is_trade[event.symbol]:
        self.__cancel_quote_levels_update((event.symbol, delta[2] % 2), delta[-1])
      self.__last_is_trade[event.symbol] = False
      self._update_snapshots(event, delta)
    else:
      self._update_trades(event)
      statuses = self._evaluate_statuses(event)
      self.__last_is_trade[event.symbol] = True

    if self.delay != 0: # if delay, statuses are also queued
      for status in statuses:
        self.pending_statuses.append((status.at, status))
      statuses = self.__update_pending_objects(event.timestamp, self.pending_statuses)
    actions = self.simulation.trigger(event, statuses, self.memory)

    self._update_composite_metrics(event, delta)
    if len(actions) > 0:
      self._process_actions(actions)
      for action in actions:
        self._flush_output(['order-request', action.symbol, action.side], action.created, action)

  def __critical_price_change(self, symbol: str, side: int, timestamp: datetime.datetime, price_volume: np.array):
    prices = price_volume[0, :]
    orders = self.simulated_orders[(symbol, side)]
    statuses = []
    # DELETE ALL ORDERS
    # for price in prices:
    #   suborders = orders.get(price, None)
    #   if suborders is not None:
    #     for sub in suborders:
    #       statuses.append(OrderStatus.cancel(sub[0], timestamp))
    #       del self.simulated_orders_id[sub[0]]
    #     del self.simulated_orders[(symbol, side)][price]

    # FINISH ALL ORDERS
    for price in prices:
      suborders = orders.get(price, None)
      if suborders is not None:
        for sub in suborders:
          statuses.append(OrderStatus.finish(sub[0], timestamp))
          del self.simulated_orders_id[sub[0]]
        del self.simulated_orders[(symbol, side)][price]


    return statuses

  def __cancel_quote_levels_update(self, symbol_side: SymbolSide,  price_volume: np.array):
    snapshot: OrderBook = self.memory[(('orderbook', symbol_side[0]))]
    target_price = snapshot.bid_prices if symbol_side[1] == QuoteSides.BID else snapshot.ask_prices
    target_volume = snapshot.bid_volumes if symbol_side[1] == QuoteSides.BID else snapshot.ask_volumes

    for i in range(price_volume.shape[-1]):
      price = float(price_volume[0, i])
      depletion = int(price_volume[1, i]) # negative value
      items = self.simulated_orders[symbol_side][price]
      
      # todo: refactor this solution
      volume_idx = np.where(target_price == price)[0]
      if volume_idx.size > 0:
        volume_idx = volume_idx[0]
        level_volume = target_volume[volume_idx]
        # todo: refactor this solution

        for i in range(len(items)):
          order_volume_before = items[i][1]
          order_volume_after = level_volume - order_volume_before

          if order_volume_before < -depletion:
            pass
          elif order_volume_after < -depletion:
            items[i] = (items[i][0], max(0, order_volume_before + depletion), items[i][2])
          else: # randomly delete this item
            if self.random.uniform(0.0, 1.0) <= order_volume_before / level_volume:
              items[i] = (items[i][0], max(0, order_volume_before + depletion), items[i][2])


  def _price_step_cancel_order(self, event: OrderBook, option: Delta) -> List[OrderStatus]:
    statuses = []

    side = option[2] % 2 # % 2 is to transform BID-ALTER -> BID and ASK-ALTER -> ASK
    orders = self.simulated_orders[(event.symbol, side)]
    altered_side_price = event.bid_prices[0] if side == QuoteSides.BID else event.ask_prices[0]
    price_to_del = []

    for price, suborders in orders.items():
      if (side == QuoteSides.BID and altered_side_price - 2 * self.price_step[event.symbol] >= price) or \
          (side == QuoteSides.ASK and altered_side_price + 2 * self.price_step[event.symbol] <= price):
        for sub in suborders:
          statuses.append(OrderStatus.cancel(sub[0], event.timestamp))
          del self.simulated_orders_id[sub[0]]
        price_to_del.append(price)
    for price in price_to_del:
      del self.simulated_orders[(event.symbol, side)][price]

    return statuses


  def run(self, tqdm_enabled=False, notify_each=3000):
    logger.info(f'Backtest initialize run')

    if tqdm_enabled:
      total = self.reader.total()
      pbar = tqdm(self.reader, total=total)
      for idx, (row, isorderbook) in enumerate(pbar):
        self._process_event(row, isorderbook)
        if idx % notify_each == 0:
          pbar.set_description(f"Current time: {row.timestamp}")

    else:
      for row, isorderbook in self.reader:
        self._process_event(row, isorderbook)
    logger.info(f'Backtest finished run')
    statuses = self._return_unfinished_orders(row.timestamp)
    self.simulation.return_unfinished(statuses, self.memory)

  def _return_unfinished_orders(self, timestamp: datetime.datetime) -> List[OrderStatus]:
    statuses = [x[1] for x in list(self.pending_statuses)]
    statuses += [OrderStatus.cancel(x[1].id, timestamp) for x in list(self.pending_orders)]
    statuses += [OrderStatus.cancel(x.id, timestamp) for x in self.simulated_orders_id.values()]
    return statuses

  def _evaluate_statuses(self, trade: Trade) -> List[OrderStatus]:
    """
    Trade simulation unit
    :param trade:
    :return:
    """
    statuses = []

    order_side = QuoteSides.BID if trade.side == TradeSides.SELL else QuoteSides.ASK # todo: do I understand it correct?
    # todo: what about aggressive orders?
    orders = self.simulated_orders[(trade.symbol, order_side)]

    if len(orders) > 0:
      # order_id, volume_total - left, consumption - ratio
      sorted_orders: List[float, OrderState] = list(sorted(orders.items(), key=lambda x: x[0]))
      to_remove = defaultdict(list)
      for price, order_requests in sorted_orders:
        for idx, (order_id, volume_level_old, consumption) in enumerate(order_requests):
          order: OrderRequest = self.simulated_orders_id[order_id]
          if (order.side == QuoteSides.BID and order.price >= trade.price) or \
              (order.side == QuoteSides.ASK and order.price <= trade.price):
          # if order.price >= trade.price or order.price <= trade.price:

            volume_for_order = trade.volume - volume_level_old
            volume_left = max(0, -volume_for_order)
            if volume_left != 0:
              orders[order.price][idx] = (order_id, volume_left, consumption)
            else:
              consumption += float(volume_for_order) / order.volume
              if consumption >= 1.0:  # order is executed
                finished = OrderStatus.finish(order_id, trade.timestamp)
                logger.info(f"Order finished: {finished}")
                statuses.append(finished)
                to_remove[order.price].append(idx)
                del self.simulated_orders_id[order.id]
              else:
                orders[order.price][idx] = (order_id, volume_left, consumption)
                if self._notify_partial and consumption > 0:
                  partial = OrderStatus.partial(order_id, trade.timestamp, int(consumption * order.volume), volume_for_order)
                  statuses.append(partial)

      for price, idxs in to_remove.items():
        self.simulated_orders[(trade.symbol, order_side)][price] = [v for i, v in enumerate(orders[price]) if i not in idxs]
        if len(self.simulated_orders[(trade.symbol, order_side)][price]) == 0:
          del self.simulated_orders[(trade.symbol, order_side)][price]

    return statuses

  def __move_order_to_active(self, action: OrderRequest):
    symbol, side, price = action.symbol, action.side, action.price
    orderbook = self.memory[('orderbook', symbol)]  # get most recent (datetime, orderbook) and return orderbook
    if side == QuoteSides.BID:
      prices = orderbook.bid_prices
      volumes = orderbook.bid_volumes
    elif side == QuoteSides.ASK:
      prices = orderbook.ask_prices
      volumes = orderbook.ask_volumes
    else:
      raise KeyError("wrong side")

    idx = np.where(prices == price)[0]
    level_volume = int(volumes[idx] or 0) # get rid of numpy

    orders = self.simulated_orders[(symbol, side)][price]
    orders_volume = sum(map(lambda x: x[0].volume * (1.0 - x[1]),
                            map(lambda x: (self.simulated_orders_id[x[0]], x[2]),
                                orders))) # todo: add explanation
    orders.append((action.id, self._generate_initial_position() * level_volume + orders_volume, 0.0))
    self.simulated_orders_id[action.id] = action

  def __update_pending_objects(self, timestamp: datetime.datetime, objects_deque: Deque) -> List[Union[OrderRequest, OrderStatus]]:
    t = timestamp - datetime.timedelta(milliseconds=self.delay)
    objs = []
    while len(objects_deque) > 0 and t >= objects_deque[0][0]:
      objs.append(objects_deque.popleft()[1])
    return objs


  def _process_actions(self, actions: List[OrderRequest]):
    for action in actions:
      if action.command == Statuses.NEW:
        if self.delay == 0:
          self.__move_order_to_active(action)
        else:
          self.pending_orders.append((action.created, action))
      elif action.command == Statuses.CANCEL:
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
      if time_metric.filter(row):
        values = time_metric.evaluate(row)
        self._flush_output(['time-metric', 'trade', row.symbol, time_metric.name], row.timestamp, values)

  def _update_composite_metrics(self, data: Union[Trade, OrderBook], option: Optional[Delta]):
    logger.debug('Update composite units')

    for composite_metric in self.simulation.composite_metrics:
      if composite_metric.filter(data):
        value = composite_metric.evaluate(data)
        self._flush_output(['composite-metric', 'snapshot', data.symbol, composite_metric.name], data.timestamp, value)

  def _update_snapshots(self, row: OrderBook, delta: Delta):
    logger.debug(f'Update units with snapshot symbol={row.symbol} @ {row.timestamp}')
    self.memory[('orderbook', row.symbol)] = row
    self._flush_output(['snapshot', row.symbol], row.timestamp, row)

    for instant_metric in self.simulation.instant_metrics:
      values = instant_metric.evaluate(row)
      self._flush_output(['instant-metric', 'snapshot', row.symbol, instant_metric.name], row.timestamp, values)

    if delta[-1].size > 0: # If any update occured
      for delta_metric in self.simulation.delta_metrics:
        if delta_metric.filter(delta):
          values = delta_metric.evaluate(delta)
          # self._flush_output(['delta', 'snapshot', row.symbol, delta_metric.name], row.timestamp, values)
          self._flush_output([delta_metric.name, row.symbol], row.timestamp, values)
      self._flush_output(['delta', row.symbol], row.timestamp, delta)

  def __str__(self):
    return '<Backtest with reader={}>'.format(self.reader)
