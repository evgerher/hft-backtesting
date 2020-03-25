from typing import Union, List, Dict

from backtesting.data import OrderStatus, OrderRequest
from backtesting.strategy import Strategy
from utils.data import Trade, OrderBook


class GatlingMM(Strategy):
  def __init__(self, side_volume=30000):
    super().__init__()
    self.side_volume = side_volume
    self.volumes_left = {} # add cancellation policy

  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
    if self.balance.get(row.symbol, None) is None:
      if isinstance(row, OrderBook) and self.balance.get(row.symbol, None) is None:
        # Initialize first orders
        ask_volume = min(self._get_allowed_volume(row.symbol, memory, 'ask'), self.side_volume)
        self.volumes_left[(row.symbol, 'ask')] = self.side_volume - ask_volume
        ask_order = OrderRequest.create_ask(row.ask_prices[0], ask_volume, row.symbol, row.timestamp)

        bid_volume = min(self._get_allowed_volume(row.symbol, memory, 'bid'), self.side_volume)
        self.volumes_left[(row.symbol, 'bid')] = self.side_volume - bid_volume
        bid_order = OrderRequest.create_bid(row.bid_prices[0], bid_volume, row.symbol, row.timestamp)

        return [ask_order, bid_order]
    elif self.balance.get(row.symbol, None) is not None:
      orders = []
      for status in statuses:
        order: OrderRequest = self.active_orders[status.id]

        if status.status != 'finished':
          self.volumes_left[(order.symbol, order.side)] += order.volume - order.volume_filled
        elif status.status == 'partial':
          self.volumes_left[(order.symbol, order.side)] += status.volume

        volume = min(self.volumes_left[(order.symbol, order.side)], self._get_allowed_volume(order.symbol, memory, order.side))
        if volume > 0:
          self.volumes_left[(order.symbol, order.side)] -= volume

          if order.side == 'bid':
            price = memory[('orderbook', order.symbol)].bid_prices[0]
          else:
            price = memory[('orderbook', order.symbol)].ask_prices[0]

          neworder = OrderRequest.create(price, volume, order.symbol, order.side, row.timestamp)
          orders.append(neworder)

      return orders
    return []
