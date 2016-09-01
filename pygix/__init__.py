#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging

logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("pygix.__init__")
    logger.error("pygix requires a python version >= 2.6")
    raise RuntimeError(
        "pygix requires a python version >= 2.6, now we are running: %s" %
        sys.version
        )
else:
    from transform import Transform
