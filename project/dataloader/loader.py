import time

from dataloader.callbacks.connectors import ClickHouse
from dataloader.utils.data import Bitmex_Data
from dataloader.callbacks.bitmex import BitmexWS
import signal

import sys
import getopt

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

finished = False

def main():
  global finished
  # kdb_connector = KDB_Connector()
  # kdb_connector.setDaemon(True)
  # kdb_connector.start()

  clickhouse_connector = ClickHouse()
  logging.info("Start app")

  dataprocessor = Bitmex_Data(clickhouse_connector)
  # .BETHXBT
  # trade
  bot = BitmexWS(
    (
      'orderBookL2_25:XBTUSD',
      'orderBookL2_25:ETHUSD',
      'trade:.BETHXBT',
      'trade:XBTUSD',
      'trade:ETHUSD'
    ), dataprocessor.callback)
  # bot = BitmexWS(('trade:.BETHXBT',), kdb.callback)
  bot.connect()

  def alarm_received(n, stack):
    # kdb_connector.close()
    global finished
    finished = True
    bot.close()
    print('SIGNAL RECEIVED')

  signal.signal(signal.SIGALRM, alarm_received)

  try:
    while not finished:
      time.sleep(1)
  finally:
    logging.info("Closing client")
    bot.close()
    finished = True
    # kdb_connector.close()
    # Stop app with kill -14 <pid>

if __name__ == '__main__':
  # Get command line parameters
  if len(sys.argv) > 1:
    try:
      opts, args = getopt.getopt(sys.argv[1:], "", ["help"])
    except getopt.GetoptError:
      print(
        'Usage: loader.py [--help]')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("--help"):
        print(
          'Dataloader starts to retrieve snapshots of BTCUSD and ETHUSD on startup from BITMEX exchange\n',
          'Stores snapshots on each delta received into kdb+\n',
          'Must be run with `pyq` interpreter.'
        )
        sys.exit(0)
  # Start load script
  main()
