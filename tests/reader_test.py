import unittest

from backtesting.readers import Reader, SnapshotReader


class ReaderTest(unittest.TestCase):
  def test_read_csv(self):
    stop_after = 100
    snapshotReader: Reader = SnapshotReader('resources/snapshots.csv', stop_after=stop_after)

    snapshots = []
    for row in snapshotReader:
      snapshots.append(row)

    self.assertEqual(len(snapshots), stop_after)

    self.assertEqual(snapshots[0].market, 'XBTUSD')
    self.assertTrue(325363 in snapshots[0].ask_volumes)

    self.assertEqual(snapshots[-1].market, 'ETHUSD')
    self.assertTrue(81742 in snapshots[-1].ask_volumes)

  def test_read_csv_gz(self):
    stop_after = 100
    snapshotReader: Reader = SnapshotReader('resources/snapshots.csv.gz', stop_after=stop_after)

    snapshots = []
    for row in snapshotReader:
      snapshots.append(row)

    self.assertEqual(len(snapshots), stop_after)
