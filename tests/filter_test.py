import unittest

import sample_reader
import numpy as np

from metrics.filters import Filters


class FilterTest(unittest.TestCase):
  def test_one_snapshot(self):
    filter = Filters.LevelFilter(3)

    snapshot = sample_reader.get_snapshots(1)[0]

    self.assertTrue(filter.filter(snapshot))

  def test_filter_memory(self):
    filter = Filters.LevelFilter(3)

    snapshots = sample_reader.get_snapshots(5)

    filtered = list(map(filter.filter, snapshots))

    self.assertEqual(len(filter.snapshots), 2)

  def test_filter_update(self):
    _filter = Filters.LevelFilter(4)
    snapshots = sample_reader.get_snapshots(10, 'resources/snapshots_filter.csv')

    filtered = list(filter(lambda x: x[0] == True, map(lambda x: (_filter.filter(x), x), snapshots)))


    self.assertTrue(len(filtered) >= 4)

