import logging


def setup_logger(name:str = None):
  # Prints logger info to terminal
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)  # Change this to DEBUG if you want a lot more info
  ch = logging.StreamHandler()
  # create formatter
  formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  # add formatter to ch
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  return logger