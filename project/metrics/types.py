import datetime
from typing import Tuple, Callable, List

from utils.data import Trade

NamedExecutable =  Tuple[str, Callable[[List], float]]
TradeExecutable = Tuple[str, Callable[[List[Trade]], float]]
DeltaValue = Tuple[datetime.datetime, int]
DeltaExecutable = Tuple[str, Callable[[List[DeltaValue]], float]]
Delta = Tuple[datetime.datetime, str, str, int]