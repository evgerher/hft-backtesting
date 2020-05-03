import datetime
import enum
from typing import Tuple, Callable, List
import numpy as np

from hft.utils.consts import QuoteSides
from hft.utils.data import Trade

NamedExecutable =  Tuple[str, Callable[[List], float]]
TradeExecutable = Tuple[str, Callable[[List[Trade]], float]]
DeltaValue = Tuple[datetime.datetime, int]
DeltaExecutable = Tuple[str, Callable[[List[DeltaValue]], float]]
Delta = Tuple[datetime.datetime, str, int, np.array]

SymbolSide = Tuple[str, int] # (symbol, side)
OrderState = Tuple[int, float, float] # (id, volume_total-left, consumption-ratio


class DepleshionReplenishmentSide(enum.Enum):
  BID_ASK = 1
  ASK_BID = 2

  @staticmethod
  def eval(sign: int, quote_side: int) -> 'DepleshionReplenishmentSide':
    return DepleshionReplenishmentSide.BID_ASK if sign > 0 and quote_side % 2 == QuoteSides.ASK or sign < 0 and quote_side % 2 == QuoteSides.BID else DepleshionReplenishmentSide.ASK_BID