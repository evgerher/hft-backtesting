from hft.backtesting.output import Output
from hft.backtesting.readers import OrderbookReader
from hft.backtesting.strategy import Strategy
from hft.backtesting.data import OrderStatus, OrderRequest
from hft.utils.consts import Statuses, QuoteSides, TradeSides
from hft.utils.data import OrderBook, Trade, Delta
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
  # todo: move part of logic into sepa rate class and extend from it (trait-like architecture)

  def __init__(self, reader: OrderbookReader,
               strategy: Strategy,
               output: Optional[Output] = None,
               order_position_policy: str = 'tail',  # 'random' or 'head'
               seed=1337,
               notify_partial: bool = True,
               delay: int=0,
               warmup: bool=False,
               stale_depth: int = 2):
    """

    :param reader:
    :param strategy:
    :param output:
    :param order_position_policy:
    :param seed:
    :param delay: delay in microseconds !
    :param warmup: whether to run backtest without strategy decisions until all time metrics are initialized or not.
    :param stale_depth: defines number of levels deep in orderbook for simulated orders to be considered stale and cancelled automatically
    """
    self.reader: OrderbookReader = reader
    self.strategy: Strategy = strategy

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

    self.stale_depth = stale_depth
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

    if warmup: # define amount of seconds for backtest to run before making decisions
      metrics = [item for sublist in self.strategy.time_metrics.values() for item in sublist]
      w = max(map(lambda metric: metric.seconds, metrics)) + 3
      self.warmup = self.reader.initial_moment + datetime.timedelta(seconds=w)
      self.warmup_ended = False
    else:
      self.warmup_ended = True
    logger.info(f"Initialized {self}")

  def _process_event(self, event: Union[Trade, OrderBook], isorderbook: bool):
    statuses = []
    delta = None
    if isorderbook:
      delta = self.strategy.filter.process(event)
      if delta is None:
        return

      if self.delay != 0:
        # update pending orders, if delay passed
        pend_orders = self.__update_pending_objects(event.timestamp, self.pending_orders)
        for ord in pend_orders:
          self.__move_order_to_active(ord)
      if delta.quote_side >= 4: # is critical
        statuses = self.__critical_price_change(event.symbol, delta.quote_side % 2, event.timestamp, delta.diff)
      elif delta.quote_side >= 2: # BID-ALTER or ASK-ALTER
        statuses = self._price_step_cancel_order(event, delta, self.stale_depth)

      if delta.diff.size > 0 and delta.diff[1, 0] < 0 and not self.__last_is_trade[event.symbol]: # todo: make it disablable
        self.__cancel_quote_levels_update((event.symbol, delta.quote_side % 2), delta.diff)
      self.__last_is_trade[event.symbol] = False
      self._update_snapshots(event, delta)
    else:
      self._update_trades(event)
      statuses = self._evaluate_statuses(event)
      self.__last_is_trade[event.symbol] = True

      # it is here just because trades are rare, thus less computations
      if not self.warmup_ended and self.warmup < self.reader.current_timestamp:
        self.warmup_ended = True

    if self.delay != 0: # if delay, statuses are also queued
      for status in statuses:
        self.pending_statuses.append((status.at, status))
      statuses = self.__update_pending_objects(event.timestamp, self.pending_statuses)

    if self.warmup_ended:
      actions = self.strategy.trigger(event, statuses, self.memory, not isorderbook)
    else:
      actions = []

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
    '''
    Method implements level's consumption below simulated quotes
    For example we have a quote on 1000 units standing on 50000 units level.
    (50k must be realized before simulated quote)
    - If delta comes with volume > level before quote, thus it is consumed from above quotes (not below).
    - Because quotes below have smaller volume.

    - It cannot be treated as trade, because trades appear before new delta orderbook, and their impact is evaluated before this delta
    - Additionaly, if it is a trade-before orderbook, no `cancel quote` will be evaluated here.

    :param symbol_side:
    :param price_volume:
    :return:
    '''
    snapshot: OrderBook = self.memory[('orderbook', symbol_side[0])]
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
          else:  # randomly delete this item
            if self.random.uniform(0.0, 1.0) <= order_volume_before / level_volume:
              items[i] = (items[i][0], max(0, order_volume_before + depletion), items[i][2])


  def _price_step_cancel_order(self, event: OrderBook, delta: Delta, level_depth=2) -> List[OrderStatus]:
    '''
    Method allows automatic cancelling of nonexecuted simulated quotes, which are far inside in orderbook from best levels.
    Thus removes stale orders and returns them to strategy.

    :param event:
    :param delta:
    :param level_depth:
    :return:
    '''
    statuses = []

    side = delta.quote_side % 2  # % 2 is to transform BID-ALTER -> BID and ASK-ALTER -> ASK
    orders = self.simulated_orders[(event.symbol, side)]
    altered_side_price = event.bid_prices[0] if side == QuoteSides.BID else event.ask_prices[0]
    price_to_del = []

    for price, suborders in orders.items():
      if (side == QuoteSides.BID and altered_side_price - level_depth * self.price_step[event.symbol] >= price) or \
          (side == QuoteSides.ASK and altered_side_price + level_depth * self.price_step[event.symbol] <= price):
        for sub in suborders:
          statuses.append(OrderStatus.cancel(sub[0], event.timestamp))
          del self.simulated_orders_id[sub[0]]
        price_to_del.append(price)
    for price in price_to_del:
      del self.simulated_orders[(event.symbol, side)][price]

    return statuses


  def run(self, tqdm_enabled=False, notify_each=3000):

    def reader_iterate():
      print('reader iterate')
      for row, isorderbook in self.reader:
        self._process_event(row, isorderbook)
      else:
        if self.reader.try_reset():
          row = reader_iterate()
      return row

    def tqdm_iterate():
      total = self.reader.total()
      pbar = tqdm(iter(self.reader), total=total)

      for idx, (row, isorderbook) in enumerate(pbar):
        self._process_event(row, isorderbook)
        if idx % notify_each == 0:
          pbar.set_description(f"Current time: {row.timestamp}")
      else:
        if self.reader.try_reset():
          row = tqdm_iterate()
      return row

    logger.info(f'Backtest initialize run')

    if tqdm_enabled:
      row = tqdm_iterate()
    else:
      row = reader_iterate()

    logger.info(f'Backtest finished run')
    statuses = self._return_unfinished_orders(row.timestamp)
    self.strategy.return_unfinished(statuses, self.memory)

  def _return_unfinished_orders(self, timestamp: datetime.datetime) -> List[OrderStatus]:
    statuses = [x[1] for x in list(self.pending_statuses)]
    statuses += [OrderStatus.cancel(x[1].id, timestamp) for x in list(self.pending_orders)]
    statuses += [OrderStatus.cancel(x.id, timestamp) for x in self.simulated_orders_id.values()]
    return statuses

  def _evaluate_statuses(self, trade: Trade) -> List[OrderStatus]:
    """
    Trade strategy unit
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
        try:
          order = self.simulated_orders_id.pop(action.id)
          symbol, side, price = order.label()
          price_orders = self.simulated_orders[(symbol, side)][price]

          idx_to_del = None
          for idx, (id, v_b, fill) in enumerate(price_orders):
            if id == order.id:
              idx_to_del = idx
          if idx_to_del is not None:
            price_orders.pop(idx_to_del)
        except: # already deleted
          pass

  def __initialize_time_metrics(self):
    for metrics in self.strategy.time_metrics.values():
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

    for time_metric in self.strategy.time_metrics['trade']:
      if time_metric.filter(row):
        values = time_metric.evaluate(row)
        self._flush_output(['time-metric', 'trade', row.symbol, time_metric.name], row.timestamp, values)

  def _update_composite_metrics(self, data: Union[Trade, OrderBook], option: Optional[Delta]):
    logger.debug('Update composite units')

    for composite_metric in self.strategy.composite_metrics:
      if composite_metric.filter(data):
        value = composite_metric.evaluate(data)
        self._flush_output(['composite-metric', 'snapshot', data.symbol, composite_metric.name], data.timestamp, value)

  def _update_snapshots(self, row: OrderBook, delta: Delta):
    logger.debug(f'Update units with snapshot symbol={row.symbol} @ {row.timestamp}')
    self.memory[('orderbook', row.symbol)] = row
    self._flush_output(['snapshot', row.symbol], row.timestamp, row)

    for instant_metric in self.strategy.instant_metrics:
      values = instant_metric.evaluate(row)
      self._flush_output(['instant-metric', 'snapshot', row.symbol, instant_metric.name], row.timestamp, values)

    if delta.diff.size > 0: # If any update occured
      for delta_metric in self.strategy.delta_metrics:
        if delta_metric.filter(delta):
          values = delta_metric.evaluate(delta)
          self._flush_output([delta_metric.name, row.symbol], row.timestamp, values)
      self._flush_output(['delta', row.symbol], row.timestamp, delta)

  def __str__(self):
    return '<Backtest with reader={}>'.format(self.reader)


class BacktestOnSample(Backtest):
  def run(self, tqdm_enabled=False, notify_each=3000):
    for row, isorderbook in self.reader:
      self._process_event(row, isorderbook)
    statuses = self._return_unfinished_orders(row.timestamp)
    self.strategy.return_unfinished(statuses, self.memory)
