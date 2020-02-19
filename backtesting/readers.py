from dataloader.utils.data import Snapshot


class Reader:

  class Row:
    pass

  def has_next(self) -> bool:
    pass

  def read_next(self) -> Snapshot:
    pass
