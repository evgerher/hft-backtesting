import unittest

from hft.backtesting.readers import OrderbookReader


class ReaderTest(unittest.TestCase):
  def test_read_csv(self):
    stop_after = 100
    snapshotReader = OrderbookReader('resources/orderbook/orderbooks.csv.gz', stop_after=stop_after)

    snapshots = []
    for row, isorderbook in snapshotReader:
      snapshots.append(row)

    self.assertEqual(len(snapshots), stop_after)

    self.assertEqual(snapshots[0].symbol, 'XBTUSD')
    self.assertTrue(56320 in snapshots[0].ask_volumes)

    self.assertEqual(snapshots[-1].symbol, 'XBTUSD')
    self.assertTrue(15081 in snapshots[-1].ask_volumes)

  def test_read_csv_gz(self):
    stop_after = 100
    reader = OrderbookReader('resources/orderbook/orderbooks.csv.gz', stop_after=stop_after)

    snapshots = []
    for row in reader:
      snapshots.append(row)

    self.assertEqual(len(snapshots), stop_after)

if __name__ == '__main__':
  unittest.main()
