import unittest

from utils.data import Snapshot
import sample_reader
import numpy as np


class SnapshotTest(unittest.TestCase):
  def test_snapshot_is_sorted(self):
    lines = sample_reader.read_snapshots('resources/snapshots_sample.txt')
    line = lines[0].split('\t')
    date = line[0]
    symbol = line[1].replace('\\0', '')
    asks = np.array(line[2:52], dtype=np.float)
    bids = np.array(line[52:], dtype=np.float)

    snapshot: Snapshot = Snapshot.from_sides(date, symbol, bids, asks)
    is_sorted = True

    ask_price = snapshot.ask_prices[0]
    bid_price = snapshot.bid_prices[0]

    # Descending order
    for b_p in snapshot.bid_prices[1:]:
      is_sorted &= bid_price > b_p
      bid_price = b_p

    # Ascending order
    for a_p in snapshot.ask_prices[1:]:
      is_sorted &= ask_price < a_p
      ask_price = a_p

    self.assertEqual(is_sorted, True)


if __name__ == '__main__':
  unittest.main()
