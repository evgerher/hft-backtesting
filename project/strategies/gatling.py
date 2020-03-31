from typing import Union, List, Dict

from backtesting.data import OrderStatus, OrderRequest
from utils.consts import Statuses, QuoteSides
from backtesting.strategy import Strategy
from utils.data import Trade, OrderBook


class GatlingMM(Strategy):
  def __init__(self, side_volume):
    super().__init__()
    self.side_volume = side_volume
    self.volumes_left = {} # add cancellation policy

  def define_orders(self, row: Union[Trade, OrderBook],
                    statuses: List[OrderStatus],
                    memory: Dict[str, Union[Trade, OrderBook]]) -> List[OrderRequest]:
    if self.balance.get(row.symbol, None) is None:
      if type(row) == OrderBook and self.balance.get(row.symbol, None) is None:
        # Initialize first orders
        ask_volume = min(self._get_allowed_volume(row.symbol, memory, QuoteSides.ASK), self.side_volume)
        self.volumes_left[(row.symbol, QuoteSides.ASK)] = self.side_volume - ask_volume
        ask_order = OrderRequest.create_ask(row.ask_prices[0], ask_volume, row.symbol, row.timestamp)

        bid_volume = min(self._get_allowed_volume(row.symbol, memory, QuoteSides.BID), self.side_volume)
        self.volumes_left[(row.symbol, QuoteSides.BID)] = self.side_volume - bid_volume
        bid_order = OrderRequest.create_bid(row.bid_prices[0], bid_volume, row.symbol, row.timestamp)

        return [ask_order, bid_order]
    elif self.balance.get(row.symbol, None) is not None:
      orders = []
      for status in statuses:
        order: OrderRequest = self.active_orders[status.id]

        if status.status != Statuses.PARTIAL: # finished and cancel
          self.volumes_left[(order.symbol, order.side)] += order.volume - order.volume_filled
        elif status.status == Statuses.PARTIAL:
          self.volumes_left[(order.symbol, order.side)] += status.volume

      for (symbol, side), left_volume in self.volumes_left.items():
        if left_volume > 500:
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
