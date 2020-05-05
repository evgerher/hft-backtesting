from dataclasses import dataclass
from collections import defaultdict

from abc import ABC, abstractmethod

@dataclass
class MetricData:
  name: str
  symbol: str
  value: float


class Metric(ABC):

  def __init__(self, name):
    self.name = name  # map of name -> metric for dependency injection
    self.latest = defaultdict(lambda: None)

  # def evaluate(self, *args) -> 'MetricData':
  @abstractmethod
  def evaluate(self, *args):
    raise NotImplementedError

  @abstractmethod
  def label(self):
    raise NotImplementedError

  def filter(self, arg):
    return True


