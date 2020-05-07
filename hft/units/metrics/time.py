import datetime
from abc import abstractmethod
from collections.__init__ import defaultdict, deque
from typing import List, Dict, Tuple, Deque, Callable, Union

import numpy as np

from hft.units.metric import Metric
from hft.units.metrics.instant import DeltaMetric
from hft.utils.data import Trade
from hft.utils.types import NamedExecutable, Delta, TradeExecutable, DeltaExecutable


class TimeMetric(Metric):
  def __init__(self, name,
               callables: List[NamedExecutable],
               seconds=60,
               starting_moment: datetime.datetime = None,
               **kwargs):

    super().__init__(name, _default_factory=lambda: defaultdict(lambda: None), **kwargs)
    self.metric_names: List[str] = [c[0] for c in callables]
    self.seconds = seconds
    # symbol, side -> trade
    self.storage: Dict[Tuple, Deque] = defaultdict(deque)
    self._callables: List[Callable[[List], float]] = [c[1] for c in callables]
    self._from: datetime.datetime = starting_moment
    self._skip_from = False

  def filter(self, event) -> bool:
    return True

  def _skip(self, event):

    if isinstance(event, Trade):
      timestamp = event.timestamp
    else:
      timestamp = event[0]

    if (timestamp - self._from).seconds >= self.seconds:
      self._skip_from = True
    return [-1.0] * len(self._callables)


  def label(self):
    return [self.name] + self.metric_names

  @abstractmethod
  def _get_update_deque(self, event) -> Tuple[Tuple, Deque]:
    raise NotImplementedError

  def evaluate(self, event: Union[Trade, Delta]) -> List[float]:
    key, target = self._get_update_deque(event)

    if not self._skip_from:
      return self._skip(event)
    else:
      self._remove_old_values(event, target)

      values = [call(target) for call in self._callables]
      names = self.metric_names
      assert len(names) == len(values)

      for idx, name in enumerate(names):
        self.latest[event.symbol][(name, *key)] = values[idx]

      return values

  @abstractmethod
  def _remove_old_values(self, event, storage: Deque[Union[Trade, Delta]]):
    raise NotImplementedError

  def _other_cases(self, event):
    pass

  def set_starting_moment(self, moment: datetime.datetime):
    self._from = moment


class TradeMetric(TimeMetric):
  def __init__(self,
               callables: List[TradeExecutable],
               seconds=60, **kwargs):
    super().__init__(f'trade-metric-{seconds}', callables, seconds, **kwargs)

  def _remove_old_values(self, event: Trade, storage: Deque[Trade]):
    while (event.timestamp - storage[0].timestamp).seconds > self.seconds:
      storage.popleft()

  def _get_update_deque(self, event: Trade):
    key = (event.symbol, event.side)
    target: Deque[Trade] = self.storage[key]
    target.append(event)
    return key, target

  def __str__(self):
    return f'trade-time-metric:{self.seconds}'


class DeltaTimeMetric(DeltaMetric, TimeMetric):
  def __init__(self,
               seconds=60,
               callables: List[DeltaExecutable] = (('quantity', lambda x: len(x)), ('volume_total', lambda x: sum(x))),
               starting_moment: datetime.datetime = None, **kwargs):

    super().__init__(f'__delta-{seconds}', callables, seconds, starting_moment, **kwargs)
    self._time_storage = defaultdict(deque)

  def _remove_old_values(self, event: Delta, storage: Deque[Delta]):
    timestamp = event[0]
    symbol = event[1]
    side = event[2]
    volume = event[3][1, 0] # get first (price, volume) pair and take volume only
    sign = 'pos' if volume > 0 else 'neg' if volume < 0 else None
    key = (symbol, side, sign)

    time_storage = self._time_storage[key] # TODO: REFACTOR TO ANOTHER TYPE
    while len(time_storage) > 50:
      time_storage.popleft()
      storage.popleft()
    # while (event[0] - time_storage[0]).seconds > self.seconds:
    #   time_storage.popleft()
    #   storage.popleft()

  def _skip(self, event):
    timestamp = event[0]
    symbol = event[1]
    side = event[2]
    volume = event[3][1, 0] # get first (price, volume) pair and take volume only
    sign = 'pos' if volume > 0 else 'neg' if volume < 0 else None

    if len(self.storage[(symbol, side, sign)]) > 50:
      self._skip_from = True
    return [-1.0] * len(self._callables)

  def _get_update_deque(self, event: Delta):
    timestamp = event[0]
    symbol = event[1]
    side = event[2] % 2  # Transform ASK-ALTER -> ASK, BID-ALTER -> BID
    volume = np.sum(event[3][1, :]) # get first (price, volume) pair and take volume only
    sign = 'pos' if volume > 0 else 'neg' if volume < 0 else None
    volume = volume if volume > 0 else -volume

    key = (symbol, side, sign)
    target: Deque[int] = self.storage[key]
    self._time_storage[key].append(timestamp)
    target.append(volume)
    return key, target

  def __str__(self):
    return f'__delta-time-metric:{self.seconds}'
