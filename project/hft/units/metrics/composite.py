import datetime
import math
from typing import Dict, List, Tuple

import numpy as np

from hft.units.metric import Metric, MetricData
from hft.units.metrics.instant import InstantMetric
from hft.utils.data import OrderBook
from hft.utils.types import DepleshionReplenishmentSide


class CompositeMetric(InstantMetric):
  def __init__(self, name: str):
    super().__init__(name)
    self._metric_map = None

  def set_metric_map(self, metric_map: Dict[str, Metric]):
    self._metric_map = metric_map


class Lipton(CompositeMetric):
  def __init__(self, vol_name: str, volume_levels=1):
    '''
    Lipton metric, evaluvates probability of Upward movement  |

    :param vol_name: name of metric where to take p_xy (UHF vol estimator, ex.: Hoyashi-Yoshido
    :param volume_levels: to consider when taking `x` & `y`
    If levels = 1, metric becomes very unstable, maybe more levels will give stability
    '''
    super().__init__('lipton')
    self.vol_metric = vol_name
    self._first_time = True
    self.volume_levels = volume_levels

    self.__n_clip = -1 + 1e-5
    self.__p_clip = 1 - 1e-5

  def _evaluate(self, snapshot: OrderBook):
    assert self._metric_map is not None
    vol_latest = self._metric_map[self.vol_metric].latest

    p_xy = vol_latest[snapshot.symbol, DepleshionReplenishmentSide.BID_ASK.name], \
           vol_latest[snapshot.symbol, DepleshionReplenishmentSide.ASK_BID.name]
    if not p_xy[0] is None and not p_xy[1] is None:
      p_xy = np.array(p_xy)
      p_xy = np.clip(p_xy, self.__n_clip, self.__p_clip)
      x = np.sum(snapshot.bid_volumes[:self.volume_levels])
      y = np.sum(snapshot.ask_volumes[:self.volume_levels])
      sqrt_corr = np.sqrt((1 + p_xy) / (1 - p_xy))
      p = 0.5 * (1. - np.arctan(sqrt_corr * (y - x) / (y + x)) / np.arctan(sqrt_corr))
      return p
    else:
      return None

  def bidask_imbalance(self, snapshot: OrderBook):
    q_b = snapshot.bid_volumes[0]
    q_a = snapshot.ask_volumes[0]
    imbalance = (q_b - q_a) / (q_b + q_a)
    return MetricData('bidask-imbalance', snapshot.symbol, imbalance)

  def label(self):
    return [self.name]


class QuickestDetection(CompositeMetric): # todo: does not work properly
  # todo: what if h1 and h2 are functions?
  def __init__(self, h1, h2, name: str, time_horizon=600):
    super().__init__(name)
    self.h1 = h1
    self.h2 = h2
    self._max = 0.0
    self._min = float('inf')
    self._values: List[float] = []
    self._dates: List[datetime.datetime] = []
    self.time_horizon = time_horizon
    self.__divisor = 4 * math.log(2)

  def _reset_state(self, price):
    self._max = price
    self._min = price

  # https://portfolioslab.com/rogers-satchell
  def roger_satchell_volatility(self):
    o, c = self._values[0], self._values[-1]
    h, l = max(self._values), min(self._values)

    # todo: does not take into consideration size of list
    return math.sqrt(math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l * o))

  def volatility_2(self):
    h, l = max(self._values), min(self._values)
    return math.sqrt(math.log(h / l) / self.__divisor)

  def _update(self, item: Tuple[float, datetime.datetime]):
    price = item[0]
    timestamp = item[1]
    self._dates.append(timestamp)
    self._values.append(price)

    if price > self._max:
      self._max = price

    if price < self._min:
      self._min = price

    while (timestamp - self._dates[0]).seconds > self.time_horizon:
      self._dates.pop(0)
      self._values.pop(0)

  def evaluate(self, item: Tuple[float, datetime.datetime]): # todo: accepts mid-point
    price     = item[0]
    self._update(item)

    sigma = self.roger_satchell_volatility()

    # check maximum condition
    upper_trend = self._max - price < self.h1 * sigma
    down_trend = price - self._min < self.h2 * sigma

    if upper_trend or down_trend:
      self._reset_state(price)

    if upper_trend and down_trend:
      raise ArithmeticError("IT IS NOT POSSIBLE")

    if upper_trend:
      return 1

    if down_trend:
      return -1

    return 0
