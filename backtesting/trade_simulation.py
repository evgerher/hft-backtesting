from dataclasses import dataclass


@dataclass
class Order:
  price: float
  volume: int
  side: str

class Simulation:
  delay = 400e-6  # 400 microsec from intranet computer to exchange terminal
  # delay = 1e-3  # 1 msec delay from my laptop

  def place_trade(self, trade: Order):
    pass


class SimulationWithTrades(Simulation):
  pass  # todo: consider placed trade and trades arrival

class SimulationWithSnapshot(Simulation):
  pass  # todo: consider only snapshots deltas