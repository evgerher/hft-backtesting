import unittest

import sample_reader

from metrics.filters import Filters


class FilterTest(unittest.TestCase):
  def test_one_snapshot(self):
    filter = Filters.DepthFilter(3)

    snapshot = sample_reader.get_snapshots(1)[0]

    self.assertTrue(filter.process(snapshot))

  def test_filter_memory(self):
    filter = Filters.DepthFilter(3)

    snapshots = sample_reader.get_snapshots(5)

    filtered = list(map(filter.process, snapshots))

    self.assertEqual(len(filter.snapshots), 2)

  def test_filter_update(self):
    _filter = Filters.DepthFilter(4)
    snapshots = sample_reader.get_snapshots(30, 'resources/snapshots_filter.csv')

    filtered = list(filter(lambda x: x[0] == True, map(lambda x: (_filter.process(x), x), snapshots)))


    self.assertTrue(len(filtered) >= 4)

  def test_filter_depth(self):
    filter3 = Filters.DepthFilter(3)
    filter4 = Filters.DepthFilter(4)
    snapshots = sample_reader.get_snapshots(10, 'resources/snapshots_depth5.csv', length=20)
    filtered3 = list(filter(lambda x: x[0] == True, map(lambda x: (filter3.process(x), x), snapshots)))
    filtered4 = list(filter(lambda x: x[0] == True, map(lambda x: (filter4.process(x), x), snapshots)))

    self.assertEqual(len(filtered3), 2)
    self.assertEqual(len(filtered4), 3) # one XBTUSD altered on ask side level=3

