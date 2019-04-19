import logging

LOGFMT = "[%(relativeCreated)d] [%(filename)s/%(funcName)s] %(message)s"
logging.basicConfig(format=LOGFMT)
