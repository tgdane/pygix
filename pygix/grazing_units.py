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

__author__ = "Thomas Dane, Jerome Kieffer"
__contact__ = "thomasgdane@gmail.com"
__license__ = "GPLv3+"
__copyright__ = "ESRF - The European Synchrotron, Grenoble, France"
__date__ = "18/11/2014"
__status__ = "Development"
__docformat__ = "restructuredtext"

import logging
logger = logging.getLogger("pygix.grazing_units")
from numpy import pi
import types
hc = 12.398419292004204


class Enum(dict):
    """
    Simple class half way between a dict and a class, behaving as an enum
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError

    def __repr__(self, *args, **kwargs):
        if "REPR" in self:
            return self["REPR"]
        else:
            return dict.__repr__(self, *args, **kwargs)

UNDEFINED = Enum(REPR='?')

somethoughts="""
ANGULAR = Enum(REPR="angular",
               x_unit="2theta_f",
               y_unit="alpha_f",
               center="gia_center_array",
               corner="gia_corner_array",
               delta="gia_delta_array",
               scale={'deg' : (180.0/pi),
                      'rad' : 1.0}

RECIPROCAL = Enum(REPR="reciprocal",
                  x_unit="qxy",
                  y_unit="qz",
                  center="giq_center_array",
                  corner="giq_corner_array",
                  delta="giq_delta_array",
                  scale={'nm' : 100.0),
                         'A' : 10.0}

POLAR = Enum(REPR="angular",
             x_unit="q",
             y_unit="chi",
             center="absq_center_array",
             corner="absq_corner_array",
             delta="absq_delta_array",
             scale={'deg' : (180.0/pi),
                    'rad' : 1.0}

SECTOR

CHI

IPBOX

OPBOX


TTH = Enum(REPR="angular",
           x_unit="2theta_f",
           y_unit="alpha_f",
           center="gia_center_array",
           corner="gia_corner_array",
           delta="gia_delta_array",
           abs_center="absa_center_array",
           abs_corner="absa_corner_array",
           abs_delta="absa_delta_array",
           scale={'deg' : (180.0/pi),
                  'rad' : 1.0}
Q = Enum(REPR='reciprocal',
         x_unit='qxy

"""

TTH_DEG = TTH = Enum(REPR="2theta_deg",
                     IPL_UNIT="2theta_f_deg",
                     OPL_UNIT="alpha_f_deg",
                     center="gia_center_array",
                     corner="gia_corner_array",
                     delta="gia_delta_array",
                     abs_center="absa_center_array",
                     abs_corner="absa_corner_array",
                     abs_delta="absa_delta_array",
                     scale=180.0/pi)

TTH_RAD = Enum(REPR="2theta_rad",
               IPL_UNIT="2theta_f_rad",
               OPL_UNIT="alpha_f_rad",  
               center="gia_center_array",
               corner="gia_corner_array",
               delta="gia_delta_array",
               abs_center="absa_center_array",
               abs_corner="absa_corner_array",
               abs_delta="absa_delta_array",     
               scale=1.0)

Q_NM = Q = Enum(REPR="q_nm^-1",
                IPL_UNIT="qxy_nm^-1",
                OPL_UNIT="qz_nm^-1",
                center="giq_center_array",
                corner="giq_corner_array",
                delta="giq_delta_array",
                abs_center="absq_center_array",
                abs_corner="absq_corner_array",
                abs_delta="absq_delta_array",
                scale=100.0)

Q_A = Enum(REPR="q_A^-1",
           IPL_UNIT="qxy_A^-1",
           OPL_UNIT="qz_A^-1",
           center="giq_center_array",
           corner="giq_corner_array",
           delta="giq_delta_array",
           abs_center="absq_center_array",
           abs_corner="absq_corner_array",
           abs_delta="absq_delta_array",
           scale=10)

GI_UNITS = (TTH_DEG, TTH_RAD, Q_NM, Q_A)

def to_unit(obj):
    giUnit = None
    if isinstance(obj, str):
        for one_unit in GI_UNITS:
            if one_unit.REPR == obj:
                giUnit = one_unit
                break
    elif obj.__class__.__name__.split(".")[-1] == "Enum":
        giUnit = obj
    if giUnit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s."
                     " Valid units are 2th_deg, 2th_rad, q_nm^-1 and q_A^-1" % \
                      (obj, type(obj)))
    return giUnit

def ip_unit(obj):
    ipl_giUnit = None
    if type(obj) in types.StringTypes:
        for one_unit in GI_UNITS:
            if one_unit.IPL_UNIT == obj:
                ipl_giUnit = one_unit
                break
    elif obj.__class__.__name__.split(".")[-1] == "Enum":
        ipl_giUnit = obj
    if ipl_giUnit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s."
                     " Valid units are 2th_deg, 2th_rad, q_nm^-1 and q_A^-1" % \
                     (obj, type(obj)))
    return ipl_giUnit  

def op_unit(obj):
    opl_giUnit = None
    if type(obj) in types.StringTypes:
        for one_unit in GI_UNITS:
            if one_unit.OPL_UNIT == obj:
                opl_giUnit = one_unit
                break
    elif obj.__class__.__name__.split(".")[-1] == "Enum":
        opl_giUnit = obj
    if opl_giUnit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s."
                     " Valid units are 2th_deg, 2th_rad, q_nm^-1 and q_A^-1" % \
                     (obj, type(obj)))
    return opl_giUnit  


# ABS_TTH_DEG = ABS_TTH = Enum(REPR="2th_deg",
#                      center="chi_tth_center_array",
#                      corner="chi_tth_corner_array",
#                      delta="chi_tth_delta_array",
#                      scale=180.0 / pi)

# ABS_TTH_RAD = Enum(REPR="2th_rad",
#                center="chi_tth_center_array",
#                corner="chi_tth_corner_array",
#                delta="chi_tth_delta_array",
#                scale=1.0)

# ABS_Q_NM = ABS_Q = Enum(REPR="q_nm^-1",
#                 center="chi_q_center_array",
#                 corner="chi_q_corner_array",
#                 delta="chi_q_delta_array",
#                 scale=100.0)

# ABS_Q_A = Enum(REPR="q_A^-1",
#            center="chi_q_center_array",
#            corner="chi_q_corner_array",
#            delta="chi_q_delta_array",
#            scale=10.0)

# ABS_UNITS = (ABS_TTH_DEG, ABS_TTH_RAD, ABS_Q_NM, ABS_Q_A)

def absolute_unit(obj):
    abs_giUnit = None
    print(type(obj))
    if type(obj) in types.StringTypes:
        for one_unit in ABS_UNITS:
            if one_unit.REPR == obj:
                abs_giUnit = one_unit
                break
    elif obj.__class__.__name__.split(".")[-1] == "Enum":
        abs_giUnit = obj
    if abs_giUnit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s. "
                     "Valid units are 2th_deg, 2th_rad, q_nm^-1, q_A^-1 and r_mm"\
                     % (obj, type(obj)))
    print(abs_giUnit)
    return abs_giUnit        

