from typing import List


def read_snapshots(src: str) -> List[str]:
  with open(src, 'r') as f:
    return f.read().split('\n')
