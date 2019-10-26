import websocket
import threading
import json
import utils
import logging

# logging = utils.setup_logging()

class BitmexWS:
  def __init__(self, topics, message_callback=None):
    # orderBookL2_25:XBTUSD, orderBookL2_25:ETHUSD
    self._topics = ','.join(topics)

    ws = websocket.WebSocketApp(f"wss://www.bitmex.com/realtime?subscribe={self._topics}",
                                on_close=self._on_close,
                                on_message=self._on_message)
    self.message_callback = message_callback
    self.ws = ws
    logging.info("Initilized BitmexWS")

  def _on_close(self, ws):
    logging.info("WS app closed")

  def _on_message(self, ws, msg):
    msg_dict = json.loads(msg)
    # logging.debug(msg)
    self.message_callback(msg_dict)


  def connect(self):
    self.thread = threading.Thread(target=self.ws.run_forever, name='WS daemon', daemon=True)
    self.thread.start()

  def close(self):
    self.ws.close()

  def subscribe(self, topic):
    # topic=orderBookL2_25:XBTUSD
    dict = {'op': 'subscribe', 'args': [topic]}
    self.ws.send(json.dumps(dict))


  def unsubsribe(self, topic):
    # topic=orderBookL2_25:XBTUSD
    dict = {'op': 'unsubscribe', 'args': [topic]}
    self.ws.send(json.dumps(dict))