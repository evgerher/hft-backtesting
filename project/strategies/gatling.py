from typing import Union, List, Dict

from backtesting.data import OrderStatus, OrderRequest
from backtesting.strategy import Strategy
from utils.data import Trade, OrderBook


class GatlingMM(Strategy):
  def __init__(self, side_volume=30000, delay=0.3):
    super().__init__(delay=delay)
    self.side_volume = side_volume
    self.volumes_set = {}

  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
    if isinstance(row, OrderBook) and self.balance.get(row.symbol, None) is None:
      # Initialize first orders
      ask_volume = self._get_allowed_volume(row.symbol, memory, 'ask')
      self.volumes_set[(row.symbol, 'ask')] = self.side_volume - ask_volume
      ask_order = OrderRequest.create_ask(row.ask_prices[0], ask_volume, row.symbol, row.timestamp)

      bid_volume = self._get_allowed_volume(row.symbol, memory, 'bid')
      self.volumes_set[(row.symbol, 'bid')] = self.side_volume - bid_volume
      bid_order = OrderRequest.create_bid(row.bid_prices[0], bid_volume, row.symbol, row.timestamp)

      return [ask_order, bid_order]
    else:
      orders = []
      for status in statuses:
        order: OrderRequest = self.active_orders[status.id]

        if status.status != 'partial':
          self.volumes_set[(row.symbol, order.side)] += order.volume - order.volume_filled
        elif status.status == 'partial':
          self.volumes_set[(row.symbol, order.side)] += status.volume

        volume = min(self.volumes_set[(row.symbol, order.side)], self._get_allowed_volume(order.symbol, memory, order.side))
        self.volumes_set[(row.symbol, 'ask')] -= volume

        if order.side == 'bid':
          price = memory[('orderbook', order.symbol)].bid_prices[0]
        else:
          price = memory[('orderbook', order.symbol)].ask_prices[0]

        neworder = OrderRequest.create(price, volume, order.symbol, order.side, row.timestamp)
        orders.append(neworder)

      return orders
