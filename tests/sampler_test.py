import unittest
from hft.environment import sampler
import shutil
import pandas as pd


class SamplerTest(unittest.TestCase):
  def test_time(self):
    dest_folder = 'time_sampled'
    samplerr = sampler.TimeSampler('resources/orderbook/orderbooks.csv.gz',
                                   'resources/orderbook/trades.csv.gz',
                                   dest_folder, 120, nrows=45000)
    samplerr.split_samples()
    df1 = pd.read_csv(f'{dest_folder}/orderbook_0.csv.gz', header=None)
    df2 = pd.read_csv(f'{dest_folder}/trade_0.csv.gz', header=None)
    ts1 = pd.to_datetime(df1[0]) # get timestamp column for orderbooks
    ts2 = pd.to_datetime(df2[1]) # get timestamp column for trades
    delta1 = abs(ts1[0] - ts2[0]).total_seconds()
    delta2 = abs(ts1[len(ts1) - 1] - ts2[len(ts2) - 1]).total_seconds()
    self.assertTrue(delta1 <= 2.) # shift
    self.assertTrue(delta2 <= 0.3) # no shift

    df1 = pd.read_csv(f'{dest_folder}/orderbook_1.csv.gz', header=None)
    df2 = pd.read_csv(f'{dest_folder}/trade_1.csv.gz', header=None)
    ts1 = pd.to_datetime(df1[0])  # get timestamp column for orderbooks
    ts2 = pd.to_datetime(df2[1])  # get timestamp column for trades
    delta1 = abs(ts1[0] - ts2[0]).total_seconds()
    delta2 = abs(ts1[len(ts1) - 1] - ts2[len(ts2) - 1]).total_seconds()
    self.assertTrue(delta1 <= 0.3)  # shift
    self.assertTrue(delta2 <= 0.3)  # no shift

    shutil.rmtree(dest_folder)

  def test_volume(self):
    dest_folder = 'volume_sampled'
    volume = 500000
    samplerr = sampler.VolumeSampler('resources/orderbook/orderbooks.csv.gz',
                                   'resources/orderbook/trades.csv.gz',
                                     dest_folder, volume, 'XBTUSD', nrows=45000)
    samplerr.split_samples()

    df1 = pd.read_csv(f'{dest_folder}/orderbook_0.csv.gz', header=None)
    df2 = pd.read_csv(f'{dest_folder}/trade_0.csv.gz', header=None)
    ts1 = pd.to_datetime(df1[0])  # get timestamp column for orderbooks
    ts2 = pd.to_datetime(df2[1])  # get timestamp column for trades
    delta1 = abs(ts1[0] - ts2[0]).total_seconds()
    delta2 = abs(ts1[len(ts1) - 1] - ts2[len(ts2) - 1]).total_seconds()
    volume_traded = df2[df2[0] == 'XBTUSD'][4].sum()

    self.assertTrue(delta1 <= 2.5)  # shift
    self.assertTrue(delta2 <= 0.3)  # no shift
    self.assertTrue(volume_traded - volume < 200000)

    df1 = pd.read_csv(f'{dest_folder}/orderbook_1.csv.gz', header=None)
    df2 = pd.read_csv(f'{dest_folder}/trade_1.csv.gz', header=None)
    ts1 = pd.to_datetime(df1[0])  # get timestamp column for orderbooks
    ts2 = pd.to_datetime(df2[1])  # get timestamp column for trades
    delta1 = abs(ts1[0] - ts2[0]).total_seconds()
    delta2 = abs(ts1[len(ts1) - 1] - ts2[len(ts2) - 1]).total_seconds()
    volume_traded = df2[df2[0] == 'XBTUSD'][4].sum()

    self.assertTrue(delta1 <= 0.3)  # small shift
    self.assertTrue(delta2 <= 1.)  # little bigger delta
    self.assertTrue(volume_traded - volume < 200000)

    shutil.rmtree(dest_folder)

  @unittest.skip('')
  def test_custom(self):
    samplerr = sampler.TimeSampler('resources/may1/orderbooks/0.csv.gz',
                                   'resources/may1/trades/0.csv.gz',
                                   'time-sampled', 300, nrows=1000000)
    samplerr.split_samples()
