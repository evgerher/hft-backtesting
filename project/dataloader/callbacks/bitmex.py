import websocket
import threading
import json
from utils.logger import setup_logger

logger = setup_logger('<orderbook10 initializer>')


class BitmexWS:
  def __init__(self, topics, message_callback=None):
    self._topics = ','.join(topics)

    self.ws = self.build_ws()
    self.message_callback = message_callback
    logger.info("Initilized BitmexWS")

  def build_ws(self):
    return websocket.WebSocketApp(f"wss://www.bitmex.com/realtime?subscribe={self._topics}",
                                  on_close=self._on_close,
                                  on_message=self._on_message)

  def _on_close(self, ws):
    logger.info("WS app closed")
    self.ws = self.build_ws()
    # self.connect()

  def _on_message(self, ws, msg):
    msg_dict = json.loads(msg)
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
