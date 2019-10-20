from time import sleep

from KDB import KDB_Bitmex, KDB_Connector
from bitmex import BitmexWS

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

def save_logs_kdb(msg: dict):
  pass

def main():
  kdb_connector = KDB_Connector()
  kdb = KDB_Bitmex(kdb_connector)

  bot = BitmexWS(('orderBookL2_25:XBTUSD','orderBookL2_25:ETHUSD',), kdb.callback)
  bot.connect()

  try:
    while True:
      sleep(1)
  except KeyboardInterrupt:
    bot.close()
    global update_f
    update_f.close()

if __name__ == '__main__':
  main()
