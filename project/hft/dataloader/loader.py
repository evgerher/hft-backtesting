import logging.config
import time

from hft.dataloader.callbacks.clickhouse.clickhouse_connector import ClickHouse
from hft.dataloader.data_processor import Bitmex_Data
from hft.utils.logger import setup_logger
from hft.dataloader.callbacks.bitmex import BitmexWS

import sys
import getopt

logger = setup_logger('<dataloader>')

finished = False

def main(db_host, db_password):
  global finished
  # kdb_connector = KDB_Connector()
  # kdb_connector.setDaemon(True)
  # kdb_connector.start()

  clickhouse_connector = ClickHouse(db_host=db_host, db_pwd=db_password)
  logger.info("Start app")

  dataprocessor = Bitmex_Data(clickhouse_connector)
  # .BETHXBT
  # trade
  bot = BitmexWS(
    (
      'orderBook10:XBTUSD',
      'orderBook10:ETHUSD',
      'trade:.BETHXBT',
      'trade:XBTUSD',
      'trade:ETHUSD',
    ), dataprocessor.callback)
  # bot = BitmexWS(('trade:.BETHXBT',), kdb.callback)
  bot.connect()

  try:
    while not finished:
      time.sleep(1)
  finally:
    logger.info("Closing client")
    bot.close()
    finished = True
    # kdb_connector.close()
    # Stop app with kill -14 <pid>
    logging.config.stopListening()

if __name__ == '__main__':
  # Get command line parameters
  if len(sys.argv) > 1:
    try:
      opts, args = getopt.getopt(sys.argv[1:], "", ["help", "host=", "password="])
      # Start load script
      opts = {x: y for (x, y) in opts}
      host, pwd = opts['--host'], opts['--password']
      main(host, pwd)
    except getopt.GetoptError:
      print(
        'Usage: loader.py [--help]')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("--help"):
        print(
          'Dataloader starts to retrieve snapshots of BTCUSD (XBTUSD) and ETHUSD on startup from BITMEX exchange\n',
          'Stores snapshots on each delta received into clickhouse server',
        )
        sys.exit(0)
