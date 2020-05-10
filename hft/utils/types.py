import enum
from typing import Tuple, Callable, List

from hft.utils.consts import QuoteSides
from hft.utils.data import Trade, DeltaValue

NamedExecutable =  Tuple[str, Callable[[List], float]]
TradeExecutable = Tuple[str, Callable[[List[Trade]], float]]
DeltaExecutable = Tuple[str, Callable[[List[DeltaValue]], float]]

SymbolSide = Tuple[str, int] # (symbol, side)
OrderState = Tuple[int, float, float] # (id, volume_total-left, consumption-ratio


class DepleshionReplenishmentSide(enum.Enum):
  BID_ASK = 1
  ASK_BID = 2

  @staticmethod
  def eval(sign: int, quote_side: int) -> 'DepleshionReplenishmentSide':
    if sign > 0 and quote_side % 2 == QuoteSides.ASK or sign < 0 and quote_side % 2 == QuoteSides.BID:
      return DepleshionReplenishmentSide.BID_ASK
    else:
      return DepleshionReplenishmentSide.ASK_BID
