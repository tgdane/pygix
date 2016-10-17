#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import absolute_import, print_function, division

__authors__ = ["Thomas Dane", "Jérôme Kieffer"]
__license__ = "GPLv3+"
__date__ = "18/03/2016"

import sys
import logging
import os

project = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

try:
    from ._version import __date__ as date  # noqa
    from ._version import version, version_info, hexversion, strictversion  # noqa
except ImportError:
    raise RuntimeError("Do NOT use %s from its sources: build it and use the built version" % project)

logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("pygix.__init__")
    logger.error("pygix requires a python version >= 2.6")
    raise RuntimeError(
        "pygix requires a python version >= 2.6, now we are running: %s" %
        sys.version
        )
else:
    from .transform import Transform
