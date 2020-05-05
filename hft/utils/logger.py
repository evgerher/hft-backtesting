import logging
import logging.config

fmt_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
to_file = False

def set_fmt_string(s):
  global fmt_string
  fmt_string = s

def setup_logger(name:str = None, level='INFO'):
  # with open('logging_config.json', 'r') as f:
  #   config = json.load(f)
  # # logging.config.fileConfig('logging_config.conf')
  # logging.config.dictConfig(config)
  logger = logging.getLogger(name)

  logger.setLevel(level)  # Change this to DEBUG if you want a lot more info
  global to_file
  if to_file:
    # if os.path.exists('logs/abcde.log'):
    #   os.remove('logs/abcde.log')
    ch = logging.FileHandler('logs/300k.log')
  else:
    ch = logging.StreamHandler()
  # create formatter
  formatter = logging.Formatter(fmt_string)
  # add formatter to ch
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  t = logging.config.listen(13000)
  t.start()
  return logger