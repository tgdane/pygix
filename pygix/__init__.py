version = "0.1.0"
date = "2014-11"
import sys, logging
logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("pygix.__init__")
    logger.error("pygix requires a python version >= 2.6")
    raise RuntimeError("pygix requires a python version >= 2.6, now we are running: %s" % sys.version)

from transform import Transform
