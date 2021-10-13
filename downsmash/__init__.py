"""
__init__.py imports some common resources used by Downsmash, most notably
the Melee percent sign.
"""

import logging
from importlib.resources import files

import numpy as np
import cv2

# Read in percent sign
NPARR = np.frombuffer(files("downsmash.resources").joinpath("pct.png").read_bytes(), np.uint8)
PERCENT = cv2.imdecode(NPARR, 1)
PERCENT = cv2.cvtColor(PERCENT, cv2.COLOR_BGR2GRAY)

LOGFMT = "[%(relativeCreated)d] [%(filename)s/%(funcName)s] %(message)s"
logging.basicConfig(format=LOGFMT)
LOGGER = logging.getLogger(__name__)
