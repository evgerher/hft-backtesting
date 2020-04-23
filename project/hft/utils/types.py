import datetime
from typing import Tuple, Callable, List
import numpy as np

from hft.utils.data import Trade

NamedExecutable =  Tuple[str, Callable[[List], float]]
TradeExecutable = Tuple[str, Callable[[List[Trade]], float]]
DeltaValue = Tuple[datetime.datetime, int]
DeltaExecutable = Tuple[str, Callable[[List[DeltaValue]], float]]
Delta = Tuple[datetime.datetime, str, int, np.array]

SymbolSide = Tuple[str, int] # (symbol, side)
OrderState = Tuple[int, float, float] # (id, volume_total-left, consumption-ratio
