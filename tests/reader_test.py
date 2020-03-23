import unittest
from typing import List

from backtesting.readers import Reader, SnapshotReader, OrderbookReader
from utils.data import OrderBook, Trade


class ReaderTest(unittest.TestCase):
  def test_read_csv(self):
    stop_after = 100
    snapshotReader = SnapshotReader('resources/snapshots.csv', stop_after=stop_after)

    snapshots = []
    for row in snapshotReader:
      snapshots.append(row)

    self.assertEqual(len(snapshots), stop_after)

    self.assertEqual(snapshots[0].symbol, 'XBTUSD')
    self.assertTrue(26773 in snapshots[0].ask_volumes)

    self.assertEqual(snapshots[-1].symbol, 'ETHUSD')
    self.assertTrue(16030 in snapshots[-1].ask_volumes)

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

  def test_orderbook_reader(self):
    reader: Reader = OrderbookReader('resources/orderbook10/orderbook.csv.gz',
                                     trades_file='resources/orderbook10/trades.csv.gz',
                                     stop_after=10000, depth_to_load=5)

    trades: List[Trade] = []

    matches = []
    snapshot1 = None
    snapshot2 = None
    snapshots = []
    last_trade = False
    suspicious = []
    for row in reader:
      if row.symbol == 'XBTUSD':
        if isinstance(row, OrderBook):
          snapshot1 = snapshot2
          snapshot2 = row
          if last_trade and snapshot1 is not None:
            if trades[-1].belongs_to(snapshot1, snapshot2):
              matches.append((trades[-1], snapshot1, snapshot2))
            else:
              suspicious.append((trades[-1], snapshot1, snapshot2))
            last_trade = False
          snapshots.append(row)
        else:
          trades.append(row)
          last_trade = True

    matched_trades = [x[0] for x in matches]
    non_matched_trades = sorted(list(set(trades) - set(matched_trades)), key=lambda x: x.timestamp)
    print('matches={}, trades={}, non_matched_trades={}'.format(len(matches), len(trades), len(non_matched_trades)))
if __name__ == '__main__':
  unittest.main()
