import unittest
from typing import List

from backtesting.readers import Reader, SnapshotReader
from utils.data import OrderBook, Trade


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

  def test_paired_snapshot_trade(self):
    reader: Reader = SnapshotReader('resources/trade/snapshots.csv.gz', trades_file='resources/trade/trades.csv.gz', stop_after=1000)

    snapshots: List[OrderBook] = []
    trades: List[Trade] = []
    for row in reader:
      if row.symbol == 'XBTUSD':
        if isinstance(row, OrderBook):
          snapshots.append(row)
        else:
          trades.append(row)

    diffs = []
    matches = []
    for idx in range(1, len(snapshots)):
      diffs.append(snapshots[idx].diff(snapshots[idx-1], 7))
      for trade in trades:
        if trade.belongs_to(snapshots[idx], snapshots[idx - 1], 7):
          matches.append((trade, snapshots[idx - 1], snapshots[idx]))

    print('yes')


