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
__author__ = "Thomas Dane, Jerome Kieffer"
__contact__ = "thomasgdane@gmail.com"
__copyright__ = "2015-2016 ESRF - The European Synchrotron, Grenoble, France"
__date__ = "18/03/2016"
__status__ = "Development"
__docformat__ = "restructuredtext"

import logging
import numpy as np
from numpy import pi, radians, sin, cos, sqrt, arcsin, arctan2

from pyFAI.geometry import Geometry

try:
    from pyFAI.ext import bilinear
except ImportError:
    bilinear = None

try:
    from pyFAI.ext.fastcrc import crc32
except ImportError:
    from zlib import crc32

logger = logging.getLogger("pygix.grazing_geometry")


class GrazingGeometry(Geometry):
    """
    This class defines the projection of pixels in detector coordinates
    into angular or reciprocal space under in the grazing-incidence
    geometry. Additionally the pixels in detector coordinates are 
    further projected into polar coordinates for azimuthal integration 
    as in pyFAI. 

    The class is inherets from pyFAI.geometry.Geometry (Geometry). The
    basic assumptions on detector configuration are therefore as 
    defined in Geometry. The correction for detector tilt is handled in
    Geometry, refer to the pyFAI documentation for more details. 

    Briefly:
    - dim1 is the Y dimension of the image
    - dim2 is the X dimension of the image
    - dim3 is along the incoming X-ray beam

    The scattering angles are defined as:
        alpha_i  = incident angle
        alpha_f  = out-of-plane exit angle
        2theta_f = in-plane exit angle

    The q-vectors are described as: (in-plane)
        qy = orthogonal to plane defined by qx, qz (in-plane)
        qz = normal to the surface (out-of-plane)

    The total in-plane scattering vector is given by:
        qxy = sqrt(qx**2 + qy**2)

    Description of the grazing-incidence transformation:
    ----------------------------------------------------
    THIS IS NOT UPDATED...
        A pixel with coordinates (t1, t2, t3) as described in Geometry, is
        first corrected for misalignment of the surface plane (if
        applicable) as follows, where the tilt angle is "misalign":
            t1 = t2 * sin(misalign) + t1 * cos(misalign)
            t2 = t2 * cos(misalign) - t1 * sin(misalign)

        The angular coordinates are given by:
            ang1 = arctan2(t1, sqrt(t2**2 + t3**2))
            ang2 = arctan2(t2, t3)

        When the surface plane is horizontal on the detector (i.e.
        giGeometry is 0 or 2):
            alpha_f  = ang1
            2theta_f = ang2
        When the surface plane is vertical on the detector (i.e.
        giGeometry is 1 or 3):
            alpha_f  = ang2
            2theta_f = ang1

        An additional correction is applied to the exit angle alpha_f to
        account for a non-zero incident angle:
            alpha_f -= alpha_i * cos(2theta_f)

        From the scattering angles, the wavevector transfer components are
        calculated as:
            k = 2 * pi/wavelength

            qx = k * (cos(alpha_f) * cos(2theta_f) - cos(alpha_i))
            qy = k * (cos(alpha_f) * sin(2theta_f))
            qz = k * (sin(alpha_f) + sin(alpha_i))
            qxy = np.copysign(sqrt(qx**2 + qy**2))

        Refraction correction:
        ----------------------
        The refractive index of matter to X-rays is given by:
            n = 1 - delta + i*beta

        For a given incident angle (ai), the effective incident angle
        transmitted within the film (ai') is slightly smaller than ai given
        by Snell's law:
            cos(ai) = n*cos(ai')

        At a given ai (known as the critical angle, ac), the effective ai'
        becomes zero. The critical angle is determined by the delta value
        of the refractive index, n:
            ac = sqrt(2 * delta)

        Snell's law can be expanded to give:
            ai'**2 = ai**2 - 2*delta + i*beta
            ai'**2 = ai**2 - ac**2 + i*beta

    """

    def __init__(self, dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0,
                 pixel1=0, pixel2=0, splinefile=None, detector=None,
                 wavelength=None, useqx=True,
                 sample_orientation=1, incident_angle=None, tilt_angle=0):
        """
        """
        Geometry.__init__(self, dist, poni1, poni2,
                          rot1, rot2, rot3,
                          pixel1, pixel2, splinefile,
                          detector, wavelength)

        self.useqx = useqx
        self.sample_orientation = sample_orientation
        self.incident_angle = incident_angle
        self.tilt_angle = tilt_angle

        # coordinate arrays for reciprocal space transformation
        self._gia_cen = None
        self._gia_crn = None
        self._gia_del = None
        self._giq_cen = None
        self._giq_crn = None
        self._giq_del = None

        # coordinate arrays for 1d and 2d integration
        self._absa_cen = None
        self._absa_crn = None
        self._absa_del = None
        self._absq_cen = None
        self._absq_crn = None
        self._absq_del = None

        # masks for missing data
        self._transformedmask = None  # 2D, reciprocal/polar
        self._chimask = None  # 1D, I vs chi

    def reset(self):
        """
        """
        Geometry.reset(self)

        self._gia_cen = None
        self._gia_crn = None
        self._gia_del = None
        self._giq_cen = None
        self._giq_crn = None
        self._giq_del = None

        self._absa_cen = None
        self._absa_crn = None
        self._absa_del = None
        self._absq_cen = None
        self._absq_crn = None
        self._absq_del = None

        self._transformedmask = None
        self._chimask = None

    # --------------------------------------------------------------------------
    #   Geometry calculations for grazing-incidence transformations
    # --------------------------------------------------------------------------

    def calc_kf_zero(self, d1, d2, param=None):
        """
        Calculate exit wavevectors kfx0, kfy0 and kfz0 as if tilt and incident
        angle were zero.

        Args:
            d1 (scalar or array): position(s) in pixel in first dimension
            d2 (scalar or array): position(s) in pixel in second dimension
            param (list): pyFAI poni geometry parameters

        Returns:
            wavevectors (tuple of floats or arrays):
        """
        if not self.wavelength:
            raise RuntimeError(("Scattering vector cannot be calculated"
                                " without knowing wavelength !!!"))
        x, z, y = self.calc_pos_zyx(d0=None, d1=d1, d2=d2, param=param,
                                    use_cython=False)
        wavevector = 1.0e-11 * 2 * pi / self.wavelength

        dd = sqrt(x ** 2 + y ** 2 + z ** 2)
        kfx0 = wavevector * (x / dd)
        kfy0 = wavevector * (-y / dd)
        kfz0 = wavevector * (z / dd)

        return kfx0, kfy0, kfz0

    def calc_q_zero(self, d1, d2, param=None):
        """
        Calculate q wavevectors qx0, qy0 and qz0 as if tilt and incident angle
        were zero.

        Args:
            d1 (scalar or array): position(s) in pixel in first dimension
            d2 (scalar or array): position(s) in pixel in second dimension
            param (list): pyFAI poni geometry parameters

        Returns:
            wavevectors (tuple of floats or arrays):
        """
        kfx0, qy0, qz0 = self.calc_kf_zero(d1, d2, param)
        wavevector = 1.0e-11 * 2 * pi / self.wavelength
        qx0 = kfx0 - wavevector
        return qx0, qy0, qz0

    def rotate_wavevectors(self, x, y, z):
        """
        Generic function to rotate wavevectors by incident and tilt angles.
        Called to calculate q (and kf for angular calculations).

        Args:
            x (scalar or array): position(s) in vector space along x axis
            y (scalar or array): position(s) in vector space along y axis
            z (scalar or array): position(s) in vector space along z axis

        Returns:
            wavevectors (float or array): rotated wavevectors
        """
        if self.sample_orientation is None:
            raise RuntimeError(("Cannot calculate without"
                                " sample orientation defined!!!"))
        if self.incident_angle is None:
            raise RuntimeError(("Cannot calculate without"
                                " incident angle defined!!!"))

        ai = radians(self._incident_angle)
        ep = radians(self._tilt_angle)
        ep += (self._sample_orientation - 1.0) * radians(90.0)

        if (ai == 0) and (ep == 0):
            wave_x, wave_y, wave_z = x, y, z
        else:
            cos_ai = cos(ai)
            sin_ai = sin(ai)
            cos_ep = cos(ep)
            sin_ep = sin(ep)

            wave_x = x * cos_ai + z * cos_ep * sin_ai + y * sin_ep * sin_ai
            wave_y = y * cos_ep - z * sin_ep
            wave_z = z * cos_ep * cos_ai + y * cos_ai * sin_ep - x * sin_ai
        return wave_x, wave_y, wave_z

    def calc_kf_xyz(self, d1, d2, param=None):
        """
        Calculate exit wavevectors kfx, kfy, kfz corrected for tilt
        and incident angle. Used for angular calculations.

        Args:
            d1 (scalar or array): position(s) in pixel in first dimension
            d2 (scalar or array): position(s) in pixel in second dimension
            param (list): pyFAI poni geometry parameters

        Returns:
            wavevectors (tuple of floats or arrays): corrected kf (x, y, z)
        """
        if param is None:
            param = self.param

        kfx0, kfy0, kfz0 = self.calc_kf_zero(d1, d2, param)
        kfx, kfy, kfz = self.rotate_wavevectors(kfx0, kfy0, kfz0)
        return kfx, kfy, kfz

    def calc_q_xyz(self, d1, d2, param=None):
        """
        Calculate qx, qy, qz corrected for tilt and incident angle.

        Args:
            d1 (scalar or array): position(s) in pixel in first dimension
            d2 (scalar or array): position(s) in pixel in second dimension
            param (list): pyFAI poni geometry parameters

        Returns:
            wavevectors (tuple of floats or arrays): corrected q (x, y, z)
        """
        if param is None:
            param = self.param

        qx0, qy0, qz0 = self.calc_q_zero(d1, d2, param)
        qx, qy, qz = self.rotate_wavevectors(qx0, qy0, qz0)

        return qx, qy, qz

    def calc_angles(self, d1, d2, param=None):
        """
        Calculate exit angles alpha_f and 2theta_f.

        Args:
            d1 (scalar or array): position(s) in pixel in first dimension
            d2 (scalar or array): position(s) in pixel in second dimension
            param (list): pyFAI poni geometry parameters

        Returns:
            scattering angles (tuple of floats or arrays): (alpha_f, 2theta_f)
        """
        kfx, kfy, kfz = self.calc_kf_xyz(d1, d2, param)
        # kfxy = sqrt(kfx**2 + kfy**2)*np.sign(kfy) not needed if below...

        alp_f = arctan2(kfz, kfx)
        tth_f = arctan2(kfy, kfx)
        return alp_f, tth_f

    def calc_q(self, d1, d2, param=None):
        """
        Calculate corrected qz, qxy.

        Args:
            d1 (scalar or array): position(s) in pixel in first dimension
            d2 (scalar or array): position(s) in pixel in second dimension
            param (list): pyFAI poni geometry parameters

        Returns:
            wavevectors (tuple of floats or arrays): corrected q (z, xy)
        """
        qx, qy, qz = self.calc_q_xyz(d1, d2, param)
        if not self.useqx:
            return qz, qy
        else:
            qxy = sqrt(qx ** 2 + qy ** 2) * np.sign(qy)
            return qz, qxy

    def calc_q_corner(self, d1, d2):
        """
        Returns (qz, qxy) for the corner of a given pixel (or set of pixels) in
        in (0.01*nm^-1).

        Args:
            d1:
            d2:

        Returns:

        """
        return self.calc_q(d1 - 0.5, d2 - 0.5)

    def calc_angles_corner(self, d1, d2):
        """
        """
        return self.calc_angles(d1 - 0.5, d2 - 0.5)

    def giq_center_array(self, shape):
        """
        Generate an array of the given shape with (qz, qxy) for all
        elements.
        """
        if self._giq_cen is None:
            with self._sem:
                if self._giq_cen is None:
                    self._giq_cen = np.fromfunction(self.calc_q, shape,
                                                    dtype=np.float32)
        return self._giq_cen

    def gia_center_array(self, shape):
        """
        """
        if self._gia_cen is None:
            with self._sem:
                if self._gia_cen is None:
                    self._gia_cen = np.fromfunction(self.calc_angles, shape,
                                                    dtype=np.float32)
        return self._gia_cen

    def giq_corner_array(self, shape):
        """
        Note: in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n, m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            qxy[0] and qz[1]
        """
        if self._giq_crn is None:
            with self._sem:
                if self._giq_crn is None:
                    qout_crn, qin_crn = np.fromfunction(
                        self.calc_q_corner,
                        (shape[0] + 1, shape[1] + 1),
                        dtype=np.float32)
                    # N.B. swap to (ipl, opl) from here on
                    if bilinear:
                        corners = bilinear.convert_corner_2D_to_4D(2, qin_crn,
                                                                   qout_crn)
                    else:
                        corners = np.zeros((shape[0], shape[1], 4, 2),
                                           dtype=np.float32)
                        corners[:, :, 0, 0] = qin_crn[:-1, :-1]
                        corners[:, :, 1, 0] = qin_crn[1:, :-1]
                        corners[:, :, 2, 0] = qin_crn[1:, 1:]
                        corners[:, :, 3, 0] = qin_crn[:-1, 1:]
                        corners[:, :, 0, 1] = qout_crn[:-1, :-1]
                        corners[:, :, 1, 1] = qout_crn[1:, :-1]
                        corners[:, :, 2, 1] = qout_crn[1:, 1:]
                        corners[:, :, 3, 1] = qout_crn[:-1, 1:]

                    self._giq_crn = corners
        return self._giq_crn

    def gia_corner_array(self, shape):
        """
        Note: in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n, m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            qxy[0] and qz[1]
        """
        if self._gia_crn is None:
            with self._sem:
                if self._gia_crn is None:
                    aout_crn, ain_crn = np.fromfunction(
                        self.calc_angles_corner,
                        (shape[0] + 1, shape[1] + 1),
                        dtype=np.float32)
                    # N.B. swap to (ipl, opl) from here on
                    if bilinear:
                        corners = bilinear.convert_corner_2D_to_4D(2, ain_crn,
                                                                   aout_crn)
                    else:
                        corners = np.zeros((shape[0], shape[1], 4, 2),
                                           dtype=np.float32)
                        corners[:, :, 0, 0] = ain_crn[:-1, :-1]
                        corners[:, :, 1, 0] = ain_crn[1:, :-1]
                        corners[:, :, 2, 0] = ain_crn[1:, 1:]
                        corners[:, :, 3, 0] = ain_crn[:-1, 1:]
                        corners[:, :, 0, 1] = aout_crn[:-1, :-1]
                        corners[:, :, 1, 1] = aout_crn[1:, :-1]
                        corners[:, :, 2, 1] = aout_crn[1:, 1:]
                        corners[:, :, 3, 1] = aout_crn[:-1, 1:]

                    self._gia_crn = corners
        return self._gia_crn

    def giq_delta_array(self, shape):
        """
        Generate 2 3D arrays of the given shape with (i,j) with the max
        distance between the center and any corner for qz and qxy.
        
        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2 2D-arrays containing the max delta between a pixel 
        center and any corner in (qz, qxy).
        """
        if self._giq_del is None:
            with self._sem:
                if self._giq_del is None:
                    if self._giq_cen is None:
                        qout_cen, qin_cen = self.giq_center_array(shape)
                    else:
                        qout_cen, qin_cen = self._giq_cen

                    qout_delta = np.zeros([shape[0], shape[1], 4],
                                          dtype=np.float32)
                    qin_delta = np.zeros([shape[0], shape[1], 4],
                                         dtype=np.float32)
                    if self._giq_crn is not None \
                            and self._giq_crn.shape[:2] == tuple(shape):
                        for i in range(4):
                            qout_delta[:, :, i] = \
                                self._giq_crn[:, :, i, 1] - qout_cen
                            qin_delta[:, :, i] = \
                                self._giq_crn[:, :, i, 0] - qin_cen
                    else:
                        qout_crn, qin_crn = np.fromfunction(
                            self.calc_q_corner,
                            (shape[0] + 1, shape[1] + 1),
                            dtype=np.float32)
                        qout_delta[:, :, 0] = abs(qout_crn[:-1, :-1] - qout_cen)
                        qout_delta[:, :, 1] = abs(qout_crn[1:, :-1] - qout_cen)
                        qout_delta[:, :, 2] = abs(qout_crn[1:, 1:] - qout_cen)
                        qout_delta[:, :, 3] = abs(qout_crn[:-1, 1:] - qout_cen)
                        qin_delta[:, :, 0] = abs(qin_crn[:-1, :-1] - qin_cen)
                        qin_delta[:, :, 1] = abs(qin_crn[1:, :-1] - qin_cen)
                        qin_delta[:, :, 2] = abs(qin_crn[1:, 1:] - qin_cen)
                        qin_delta[:, :, 3] = abs(qin_crn[:-1, 1:] - qin_cen)

                    qout_delta = qout_delta.max(axis=2)
                    qin_delta = qin_delta.max(axis=2)

                    qin_delta[np.where(qin_delta > 5e-4)] = 0

                    self._giq_del = (qout_delta, qin_delta)
        return self._giq_del

    def gia_delta_array(self, shape):
        """
        Generate 2 3D arrays of the given shape with (i,j) with the max
        distance between the center and any corner for alpha_f and 2theta_f.
        
        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2 2D-arrays containing the max delta between a pixel 
        center and any corner in (alpha_f, 2theta_f).
        """
        if self._gia_del is None:
            with self._sem:
                if self._gia_del is None:
                    if self._gia_cen is None:
                        aout_cen, ain_cen = self.gia_center_array(shape)
                    else:
                        aout_cen, ain_cen = self._gia_cen

                    aout_delta = np.zeros([shape[0], shape[1], 4],
                                          dtype=np.float32)
                    ain_delta = np.zeros([shape[0], shape[1], 4],
                                         dtype=np.float32)
                    if self._gia_crn is not None \
                            and self._gia_crn.shape[:2] == tuple(shape):

                        for i in range(4):
                            aout_delta[:, :, i] = \
                                self._gia_crn[:, :, i, 1] - qout_cen
                            ain_delta[:, :, i] = \
                                self._gia_crn[:, :, i, 0] - qin_cen
                    else:
                        aout_crn, ain_crn = np.fromfunction(
                            self.calc_angles_corner,
                            (shape[0] + 1, shape[1] + 1),
                            dtype=np.float32)
                        aout_delta[:, :, 0] = abs(aout_crn[:-1, :-1] - aout_cen)
                        aout_delta[:, :, 1] = abs(aout_crn[1:, :-1] - aout_cen)
                        aout_delta[:, :, 2] = abs(aout_crn[1:, 1:] - aout_cen)
                        aout_delta[:, :, 3] = abs(aout_crn[:-1, 1:] - aout_cen)
                        ain_delta[:, :, 0] = abs(ain_crn[:-1, :-1] - ain_cen)
                        ain_delta[:, :, 1] = abs(ain_crn[1:, :-1] - ain_cen)
                        ain_delta[:, :, 2] = abs(ain_crn[1:, 1:] - ain_cen)
                        ain_delta[:, :, 3] = abs(ain_crn[:-1, 1:] - ain_cen)

                    self._gia_del = (
                    aout_delta.max(axis=2), ain_delta.max(axis=2))
        return self._gia_del

    # --------------------------------------------------------------------------
    #   Geometry calculations for 2d and 1d integrations
    # --------------------------------------------------------------------------

    def calc_absq_corner(self, d1, d2):
        """
        Returns (alpf, tthf) for the corner of a given pixel
        (or set of pixels) in radians.
        """
        qout, qin = self.calc_q(d1 - 0.5, d2 - 0.5)

        q = sqrt(qin ** 2 + qout ** 2)
        chi = arctan2(qin, qout) + pi
        return chi, q

    def calc_absa_corner(self, d1, d2):
        """
        Returns (alpf, tthf) for the corner of a given pixel
        (or set of pixels) in radians.
        """
        aout, ain = self.calc_angles(d1 - 0.5, d2 - 0.5)

        ang = sqrt(ain ** 2 + aout ** 2)
        chi = arctan2(ain, aout) + pi
        return chi, ang

    def absq_center_array(self, shape):
        """
        """
        qout, qin = self.giq_center_array(shape)

        q = sqrt(qin ** 2 + qout ** 2)
        chi = arctan2(qin, qout) + pi

        self._absq_cen = (chi, q)
        return self._absq_cen

    def absa_center_array(self, shape):
        """
        """
        aout, ain = self.gia_center_array(shape)

        tth = sqrt(ain ** 2 + aout ** 2)
        chi = arctan2(ain, aout) + pi

        self._absa_cen = (chi, tth)
        return self._absa_cen

    def absq_corner_array(self, shape):
        """
        N.B. in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n,m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            tthf[0] and alpf[1]
        """
        q_crn = self.giq_corner_array(shape)
        corners = np.zeros((shape[0], shape[1], 4, 2),
                           dtype=np.float32)
        corners[:, :, :, 0] = sqrt(
            q_crn[:, :, :, 0] ** 2 + q_crn[:, :, :, 1] ** 2)
        corners[:, :, :, 1] = arctan2(q_crn[:, :, :, 0], q_crn[:, :, :, 1]) + pi

        self._absq_crn = corners
        return self._absq_crn

    def absa_corner_array(self, shape):
        """
        N.B. in all other coord functions, values are returned as
        (opl, ipl). Due to requirements for splitpix method here are 
        returned (ipl, opl). 

        return : (n,m, 4, 2) array
            where n, m is the image dimensions, 
            4 is the four corners and 2 is the
            tthf[0] and alpf[1]
        """
        ang_crn = self.gia_corner_array(shape)
        corners = np.zeros((shape[0], shape[1], 4, 2),
                           dtype=np.float32)
        corners[:, :, :, 0] = sqrt(
            ang_crn[:, :, :, 0] ** 2 + ang_crn[:, :, :, 1] ** 2)
        corners[:, :, :, 1] = arctan2(ang_crn[:, :, :, 0],
                                      ang_crn[:, :, :, 1]) + pi

        self._absa_crn = corners
        return self._absa_crn

    def absq_delta_array(self, shape):
        """
        """
        if self._absq_del is None:
            with self._sem:
                if self._absq_del is None:
                    if self._absq_cen is None:
                        chi_cen, q_cen = self.absq_center_array(shape)
                    else:
                        chi_cen, q_cen = self._absq_cen
                    chi_delta = np.zeros([shape[0], shape[1], 4],
                                         dtype=np.float32)
                    q_delta = np.zeros([shape[0], shape[1], 4],
                                       dtype=np.float32)
                    if (self._absq_crn is not None) \
                            and (self._absq_crn.shape[:2] == tuple(shape)):
                        for i in range(4):
                            chi_delta[:, :, i] = \
                                self._absq_crn[:, :, i, 1] - chi_cen
                            q_delta[:, :, i] = \
                                self._absq_crn[:, :, i, 0] - q_cen
                    else:
                        chi_crn, q_crn = np.fromfunction(
                            self.calc_absq_corner,
                            (shape[0] + 1, shape[1] + 1),
                            dtype=np.float32)

                        chi_delta[:, :, 0] = abs(chi_crn[:-1, :-1] - chi_cen)
                        chi_delta[:, :, 1] = abs(chi_crn[1:, :-1] - chi_cen)
                        chi_delta[:, :, 2] = abs(chi_crn[1:, 1:] - chi_cen)
                        chi_delta[:, :, 3] = abs(chi_crn[:-1, 1:] - chi_cen)
                        q_delta[:, :, 0] = abs(q_crn[:-1, :-1] - q_cen)
                        q_delta[:, :, 1] = abs(q_crn[1:, :-1] - q_cen)
                        q_delta[:, :, 2] = abs(q_crn[1:, 1:] - q_cen)
                        q_delta[:, :, 3] = abs(q_crn[:-1, 1:] - q_cen)

                    chi_delta = chi_delta.max(axis=2)
                    q_delta = q_delta.max(axis=2)

                    chi_delta[np.where(chi_delta > 0.2e-1)] = 0
                    # q_delta[np.where(q_delta > 1e-10)] = 0

                    self._absq_del = (chi_delta, q_delta)
        return self._absq_del

    def absa_delta_array(self, shape):
        """
        """
        chi_cen, ang_cen = self.absa_center_array(shape)

        if self._absa_del is None:
            with self._sem:
                if self._absa_del is None:
                    chi_delta = np.zeros([shape[0], shape[1], 4],
                                         dtype=np.float32)
                    ang_delta = np.zeros([shape[0], shape[1], 4],
                                         dtype=np.float32)
                    if (self._absa_crn is not None) \
                            and (self._absa_crn.shape[:2] == tuple(shape)):
                        for i in range(4):
                            chi_delta[:, :, i] = \
                                self._absq_crn[:, :, i, 1] - chi_cen
                            ang_delta[:, :, i] = \
                                self._absq_crn[:, :, i, 0] - ang_cen
                    else:
                        chi_crn, ang_crn = np.fromfunction(
                            self.calc_absa_corner,
                            (shape[0] + 1, shape[1] + 1),
                            dtype=np.float32)

                        chi_delta[:, :, 0] = abs(chi_crn[:-1, :-1] - chi_cen)
                        chi_delta[:, :, 1] = abs(chi_crn[1:, :-1] - chi_cen)
                        chi_delta[:, :, 2] = abs(chi_crn[1:, 1:] - chi_cen)
                        chi_delta[:, :, 3] = abs(chi_crn[:-1, 1:] - chi_cen)
                        ang_delta[:, :, 0] = abs(ang_crn[:-1, :-1] - ang_cen)
                        ang_delta[:, :, 1] = abs(ang_crn[1:, :-1] - ang_cen)
                        ang_delta[:, :, 2] = abs(ang_crn[1:, 1:] - ang_cen)
                        ang_delta[:, :, 3] = abs(ang_crn[:-1, 1:] - ang_cen)

                    chi_delta = chi_delta.max(axis=2)
                    ang_delta = ang_delta.max(axis=2)
                    self._absa_del = (chi_delta, ang_delta)
        return self._absa_del

    # ---------------------------------------------------------------------------
    #   Masking functions for inaccessible data under grazing-incidence
    # ---------------------------------------------------------------------------

    def make_transformedmask(self, process, inshape, npt, bins_x, bins_y):
        """
        Currently not implemented, bloody mess.

        Transformation routines bbox, splitpix, csr and lut (+ocl) result in
        interpolation over the unaccessible regions of reciprocal space. This
        function makes a mask for these regions.

        This is determined by setting qy = 0 (i.e. when 2theta_f = 0) and 
        calculating qx as a function of qz:

        qx = k(cos(alpha_f)cos(2theta_f) - cos(alpha_i))
        qx = k(cos(alpha_f) - cos(alpha_i))

        qz = k(sin(alpha_f) + sin(alpha_i))
        alpha_f = arcsin(qz/k - sin(alpha_i))
        """
        wvec = 1.0e-11 * 2 * pi / self._wavelength
        ai = radians(self._incident_angle)
        cos_ai = cos(ai)
        sin_ai = sin(ai)

        # print 'inshape, npt, bins_x.shape, bins_y.shape'
        # print inshape, npt, bins_x.shape, bins_y.shape

        with self._sem:
            qx, qy, qz = np.fromfunction(self.calc_q_xyz, inshape,
                                         dtype=np.float32)

        # print 'qx.shape, qy.shape, qz.shape'
        # print qx.shape, qy.shape, qz.shape

        pp = '''
        fig = plt.figure(figsize=(15,4))
        fig.add_subplot(131)
        plt.imshow(qx*100, cmap='afmhot', origin='lower')
        plt.colorbar()
        
        fig.add_subplot(132)
        plt.imshow(qy*100, cmap='afmhot', origin='lower')
        plt.colorbar()
        
        fig.add_subplot(133)
        plt.imshow(qz*100, cmap='afmhot', origin='lower')
        plt.colorbar()
        
        plt.show()
        #'''

        mask = np.ones(npt).astype(np.float32).ravel()

        # extract a line along qx vs qz at qy = 0
        # N.B. WILL NEED TO HANDLE SAMPLE ORIENTATIONS STILL

        if self.sample_orientation in [1, 3]:
            axis = 1
        else:
            axis = 0

        zeroidx = np.argmin(abs(qy), axis=axis)  # where qy closest to 0
        y = np.linspace(0, inshape[abs(1 - axis)] - 1,
                        inshape[abs(1 - axis)])  # idx along qz

        # print 'zeroidx.shape, y.shape'
        # print zeroidx.shape, y.shape

        if self.sample_orientation in [1, 3]:
            qxqy0 = abs(qx[y.astype(np.int), zeroidx.astype(np.int)])
        else:
            qxqy0 = abs(qx[zeroidx.astype(np.int), y.astype(np.int)])

        # plt.plot(qxqy0)
        # plt.plot(abs(qx[:,786]))
        # plt.plot(abs(qx[:,813]))
        # print zeroidx
        # plt.plot(y[350:550], qxqy0[350:550], 'ko')
        # for i in range(-20, 20):
        #    ddd = abs(qx[350:550,800+i])
        #    print 800+i, ddd.min()
        #    plt.plot(ddd)
        # plt.xlim(0,500)
        # plt.show()

        if 'reciprocal' in process:
            if self.sample_orientation in [1, 3]:
                qxy = np.outer(np.ones(bins_y.shape[0]), bins_x).ravel()
                qlim = np.outer(qxqy0, np.ones(bins_x.shape[0])).ravel()

            # print qxy.shape, qlim.shape

            else:
                qxy = np.outer(bins_x, np.ones(bins_y.shape[0])).ravel()
                qlim = np.outer(np.ones(bins_x.shape[0]), qxqy0).ravel()

            mask[abs(qxy) < qlim] = 0
            mask = np.reshape(mask, (npt[1], npt[0]))

            # plt.imshow(mask)
            # plt.show()

        else:
            qzl = qz[zeroidx.astype(np.int), y.astype(np.int)]

            chilim_full = (np.arctan2(qxqy0, qzl))  # [::-1]
            qabs = np.sqrt(qxqy0 ** 2 + qzl ** 2)

            zeropoint = np.argmin(qabs)
            chilim_lower = chilim_full[zeropoint:]
            extnd = chilim_lower.shape[0] * (np.max(bins_x) / np.max(qabs) - 1)
            chilim_lower = np.append(chilim_lower, np.zeros(extnd))

            npt_chilim = chilim_lower.shape[0]
            y1 = np.linspace(0, npt_chilim - 1, npt_chilim)
            y2 = np.linspace(0, npt_chilim - 1, npt[0])  # npt[0])
            chilim = np.interp(y2, y1, chilim_lower)

            # chilim = np.append(chilim, np.zeros(npt[0]-npt_out))
            chi = np.outer(bins_y, np.ones(bins_x.shape[0])).ravel()
            chi_lim = np.outer(np.ones(bins_y.shape[0]), chilim).ravel()

            # for full mapping would use:
            #     mask[chi_lower < abs(chi) < chi_upper] = 0
            mask[abs(chi - pi) <= chi_lim] = 0

            mask = np.reshape(mask, (npt[1], npt[0]))

            # --------------------------------
            # entire code for this else clause
            this_works = '''
            qzl = qz[zeroidx.astype(np.int), y.astype(np.int)]
            
            chilim_full = (np.arctan2(qxqy0, qzl))#[::-1]
            qabs = np.sqrt(qxqy0**2 + qzl**2)
            
            zeropoint = np.argmin(qabs)
            chilim_lower = chilim_full[zeropoint:]
            
            npt_out = (round(npt[0]*(np.max(qabs)/np.max(bins_x))))#+50
            npt_chilim = chilim_lower.shape[0]
            y1 = np.linspace(0, npt_chilim-1, npt_chilim)
            y2 = np.linspace(0, npt_chilim-1, npt_out)#npt[0])
            
            
            chilim = np.interp(y2, y1, chilim_lower)
            chilim = np.append(chilim, np.zeros(npt[0]-npt_out))
            chi = np.outer(bins_y, np.ones(bins_x.shape[0])).ravel()
            chi_lim = np.outer(np.ones(bins_y.shape[0]), chilim).ravel()
            
            # for full mapping would use:
            #     mask[chi_lower < abs(chi) < chi_upper] = 0
            mask[abs(chi-pi) <= chi_lim] = 0
            
            mask = np.reshape(mask, (npt[1],npt[0]))
            #'''
            # ------------------------------

        self._transformedmask = mask
        return self._transformedmask

    def make_chimask(self, q_max, chi):
        """
        """
        mask = np.ones(chi.shape).astype(np.float32)

        alpi = radians(self._incident_angle)
        wvec = 1.0e-9 * 2 * pi / self._wavelength

        cos_alpi = cos(alpi)
        sin_alpi = sin(alpi)
        q_lim = abs(wvec * (cos(arcsin((q_max / wvec) - sin_alpi)) - \
                            cos_alpi))

        chi_lim = np.rad2deg(arctan2(q_lim, q_max))
        mask[abs(chi) < chi_lim] = 0

        self._chimask = mask
        return self._chimask

    # --------------------------------------------------------------------------
    #   Properties
    # --------------------------------------------------------------------------

    def set_useqx(self, useqx):
        if not isinstance(useqx, (bool)):
            raise RuntimeError("useqx must be True or False")
        else:
            self._useqx = useqx
        self.reset()

    def get_useqx(self):
        return self._useqx

    useqx = property(get_useqx, set_useqx)

    def set_sample_orientation(self, sample_orientation):
        if sample_orientation in range(1, 5):
            self._sample_orientation = sample_orientation
        else:
            err_msg = """
            Sample orientation must be an integeter from 1 to 4:
              1: sample plane horizontal; +ve qz = bottom-to-top
              2: sample plane vertical;   +ve qz = left-to-right
              3: sample plane horizontal; +ve qz = top-to-bottom
              4: sample plane vertical;   +ve qz = right-to-left.
            """
            raise ValueError(err_msg)
        self.reset()

    def get_sample_orientation(self):
        return self._sample_orientation

    sample_orientation = property(get_sample_orientation,
                                  set_sample_orientation)

    def set_incident_angle(self, incident_angle):
        self._incident_angle = incident_angle
        self.reset()

    def get_incident_angle(self):
        return self._incident_angle

    incident_angle = property(get_incident_angle, set_incident_angle)

    def set_tilt_angle(self, tilt_angle):
        self._tilt_angle = tilt_angle
        self.reset()

    def get_tilt_angle(self):
        return self._tilt_angle

    tilt_angle = property(get_tilt_angle, set_tilt_angle)
