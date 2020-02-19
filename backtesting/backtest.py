from backtesting.readers import Reader
from backtesting.trade_simulation import Simulation

class Backtest:

  def __init__(self, reader: Reader, simulation: Simulation):
    self.reader = reader
    self.simulation = simulation

  def run(self):
    pass
