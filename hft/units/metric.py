from dataclasses import dataclass
from collections import defaultdict, deque

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


@dataclass
class MetricData:
  name: str
  symbol: str
  value: float


class ZNormalized(defaultdict):
  '''
  Class provides automatic z-normalization over latest observations
  Amount of observations is defined in constructor
  '''
  def __init__(self, period: int, default_factory=None, **kwargs):
    '''

    :param period: amount of observations to store
    :param default_factory: factory for `defaultdict`
    '''
    super().__init__(default_factory, **kwargs)
    self.__storage = defaultdict(lambda: deque(maxlen=period))

  def __setitem__(self, item, value):
    '''
    Along with setting an item, store an observation in deque
    :return:
    '''
    super().__setitem__(item, value)
    self.__storage[item].append(value)

  def __getitem__(self, item) -> np.array:
    '''
    Returns normalized value by key
    Normalization is done via computing mu & sigma from latest observations

    :return: normalized value
    '''
    values = np.array(self.__storage[item], dtype=np.float)
    value = super().__getitem__(item)

    mu, std = np.mean(values, axis=0), np.std(values, axis=0)
    z = (value - mu) / (std + 1e-4) # to avoid division by 0
    return z

  def values(self):
    '''

    :return: normalized projected view of a dict
    '''
    return list(map(self.__getitem__, super().keys()))

  def items(self):
    return self.keys(), self.values()



class Metric(ABC):

  def __init__(self, name, z_normalize: Optional[int] = None, _default_factory=lambda: None):
    '''
    Base class for Metric object
    :param name:
    :param z_normalize:
    '''
    self.name = name  # map of name -> metric for dependency injection
    if z_normalize is None:
      self.latest = defaultdict(_default_factory)
    else:
      self.latest = ZNormalized(period=z_normalize, default_factory=_default_factory)

  # def evaluate(self, *args) -> 'MetricData':
  @abstractmethod
  def evaluate(self, *args):
    raise NotImplementedError

  @abstractmethod
  def label(self):
    raise NotImplementedError

  def filter(self, arg):
    return True

  @abstractmethod
  def to_numpy(self):
    raise NotImplementedError


