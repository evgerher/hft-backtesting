from KDB import KDB_Bitmex, KDB_Connector
from bitmex import BitmexWS
import signal

import sys
import getopt

update_f = None

def open_files():
  global update_f
  update_f = open("example/update.json", 'w+')

def save_logs_locally(text: dict):
  import json

  global update_f
  if update_f is None:
    open_files()
  update_f.write(json.dumps(text) + '\n')

def main():
  kdb_connector = KDB_Connector()
  # kdb_connector.setDaemon(True)
  # kdb_connector.start()

  kdb = KDB_Bitmex(kdb_connector)
  # .BETHXBT
  # trade
  bot = BitmexWS(('orderBookL2_25:XBTUSD','orderBookL2_25:ETHUSD','trade:.BETHXBT'), kdb.callback)
  # bot = BitmexWS(('trade:.BETHXBT',), kdb.callback)
  bot.connect()

  def alarm_received(n, stack):
    kdb_connector.close()
    bot.close()
    print('SIGNAL RECEIVED')

  signal.signal(signal.SIGALRM, alarm_received)

  try:
    kdb_connector.run()
  finally:
    bot.close()
    kdb_connector.close()
    global update_f
    if update_f is not None:
      update_f.close()

    # Stop app with kill -14 <pid>

if __name__ == '__main__':
  # Get command line parameters
  if len(sys.argv) > 1:
    try:
      opts, args = getopt.getopt(sys.argv[1:], "", ["help"])
    except getopt.GetoptError:
      print(
        'Usage: dataloader.py [--help]')
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
