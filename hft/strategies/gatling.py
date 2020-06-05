import datetime
from collections import defaultdict
from typing import Union, List, Dict

from hft.backtesting.data import OrderStatus, OrderRequest
from hft.utils.consts import Statuses, QuoteSides
from hft.backtesting.strategy import Strategy
from hft.utils.data import Trade, OrderBook


class GatlingMM(Strategy):
  def __init__(self, side_volume, filter=4, **kwargs):
    super().__init__(filter=filter, **kwargs)
    self.side_volume = side_volume
    self.volumes_left = {} # add cancellation policy
    self.settled_first = defaultdict(lambda: False)

  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]],
                    is_trade: bool) -> List[OrderRequest]:

    if row.symbol != 'ETHUSD' and not self.settled_first[row.symbol]:
      if not is_trade:
        # Initialize first orders
        ask_volume = min(self._get_allowed_volume(row.symbol, memory, QuoteSides.ASK), self.side_volume)
        self.volumes_left[(row.symbol, QuoteSides.ASK)] = self.side_volume - ask_volume
        ask_order = OrderRequest.create_ask(row.ask_prices[0], ask_volume, row.symbol, row.timestamp)

        bid_volume = min(self._get_allowed_volume(row.symbol, memory, QuoteSides.BID), self.side_volume)
        self.volumes_left[(row.symbol, QuoteSides.BID)] = self.side_volume - bid_volume
        bid_order = OrderRequest.create_bid(row.bid_prices[0], bid_volume, row.symbol, row.timestamp)
        self.settled_first[row.symbol] = True

        return [ask_order, bid_order]
    # elif self.balance.get(row.symbol, None) is not None:
    else:
      orders = []
      for status in statuses:
        order: OrderRequest = self.active_orders[status.id]

        if status.status == Statuses.FINISHED: # finished and cancel
          self.volumes_left[(order.symbol, order.side)] += order.volume - order.volume_filled
        elif status.status == Statuses.CANCEL:
          self.volumes_left[(order.symbol, order.side)] += status.volume
        elif status.status == Statuses.PARTIAL:
          self.volumes_left[(order.symbol, order.side)] += status.volume

      for (symbol, side), left_volume in self.volumes_left.items():
        if left_volume > 100:
          volume = min(left_volume, self._get_allowed_volume(symbol, memory, side))
          self.volumes_left[(symbol, side)] -= volume

          if side == QuoteSides.BID:
            price = memory[('orderbook', symbol)].bid_prices[0]
          else:
            price = memory[('orderbook', symbol)].ask_prices[0]

          neworder = OrderRequest.create(price, volume, symbol, side, row.timestamp)
          orders.append(neworder)

      return orders
    return []

  def _balance_listener(self, memory: Dict[str, Union[Trade, OrderBook]],
                        ts: datetime.datetime,
                        orders: List[OrderRequest],
                        statuses: List[OrderStatus]):
    if self.balance_listener is not None and (len(orders) > 0 or len(statuses) > 0):
      # balance = memory[('orderbook', 'XBTUSD')].bid_prices[0] * self.balance['XBTUSD'] + \
      #   memory[('orderbook', 'ETHUSD')].bid_prices[0] * self.balance['ETHUSD'] + \
      #   self.balance['USD']

      # midpoint_eth = (memory[('orderbook', 'ETHUSD')].bid_prices[0] + memory[('orderbook', 'ETHUSD')].ask_prices[0]) / 2
      midpoint_xbt = (memory[('orderbook', 'XBTUSD')].bid_prices[0] + memory[('orderbook', 'XBTUSD')].ask_prices[0]) / 2

      state = (self.balance['USD'], self.balance['XBTUSD'], midpoint_xbt, ts, *self.position['XBTUSD'])
      # self.balance_listener(self.balance['USD'], row.timestamp)
      self.balance_listener(state)
