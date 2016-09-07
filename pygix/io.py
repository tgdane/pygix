#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" input/output functions.
"""

from . import grazing_units


class Writer(object):
    """

    """

    def __init__(self, pg, filename):
        """

        Returns:

        """
        self._pg = pg
        self._filename = filename
        self._header = None
        self._already_written = False

    def make_header(self, hdr="#", has_dark=False, has_flat=False,
                    polarization_factor=None, normalization_factor=None):
        """
        Make headers for data file.

        Args:
            hdr (string): string used as comment in the header.
            has_dark (bool): save the file names of dark images.
            has_flat (bool): save the file names of flat images.
            polarization_factor (float):
            normalization_factor (float):

        Returns:
            self._header (string):
        """
        if self._header is None:
            pg = self._pg
            header_list = ["== pyFAI calibration =="]
            header_list.append("Spline file: %s" % pg.splineFile)
            header_list.append("Pixel size: %.3e, %.3e m" %
                               (pg.pixel1, pg.pixel2))
            header_list.append("PONI: %.3e, %.3e m" % (pg.poni1, pg.poni2))
            header_list.append("Sample-detector distance: %s m" %
                               pg.dist)
            header_list.append("Rotations: %.6f, %.6f, %.6f rad" %
                               (pg.rot1, pg.rot2, pg.rot3))
            if pg._wavelength is not None:
                header_list.append("Wavelength: %s" % pg.wavelength)
            header_list.append("== pygix parameters ==")
            header_list.append("Sample orientation: %g" % pg.sample_orientation)
            header_list.append("Incident angle: %.6f" % pg.incident_angle)
            header_list.append("Tilt angle: %.6f" % pg.tilt_angle)

            header_list.append("== corrections ==")
            if pg.maskfile is not None:
                header_list.append("Mask file: %s" % pg.maskfile)
            if has_dark or (pg.darkcurrent is not None):
                if pg.darkfiles:
                    header_list.append("Dark current: %s" % pg.darkfiles)
                else:
                    header_list.append("Dark current: Done with unknown file")
            if has_flat or (pg.flatfield is not None):
                if pg.flatfiles:
                    header_list.append("Flat field: %s" % pg.flatfiles)
                else:
                    header_list.append("Flat field: Done with unknown file")
            if (polarization_factor is None) and (pg._polarization is not None):
                polarization_factor = pg._polarization_factor
            header_list.append("Polarization factor: %s" % polarization_factor)
            header_list.append(
                "Normalization factor: %s" % normalization_factor)
            self._header = "\n".join([hdr + " " + i for i in header_list])

        return self._header

    # def save_2d(self, filename, I, dim1, dim2,
    #             error=None, dim1_unit=grazing_units.Q,
    #             dark=None, flat=None, polarization_factor=None,
    #             normalization_factor=None):
    #     """
    #     This method saves the result of 2D integration. If no filename is
    #     given the function will exit.
    #
    #     Parameters
    #     ----------
    #     filename : str
    #         The file name to save the transformed image.
    #     I : ndarray
    #         Transformed image (intensity).
    #     dim1 : ndarray
    #         The 1st coordinates of the histogram (radial).
    #     dim2 : ndarray
    #         The 2nd coordinates of the histogram (azimuthal).
    #     error : ndarray or None
    #         The error bar for each intensity.
    #     dim1_unit : PyGIX.grazing_units.Unit
    #         Unit of the dim1 array.
    #     dark :  ???
    #         Save the dark filename(s) (default: no).
    #     flat : ???
    #         Save the flat filename(s) (default: no).
    #     polarization_factor : float
    #         The polarization factor.
    #     normalization_factor: float
    #         The normalization factor.
    #     giGeometry : int
    #         Grazing-incidence geometry in [0, 1, 2, 3].
    #     alpha_i : float
    #         Incident angle.
    #     misalign : float
    #         Surface plane tilt angle.
    #     """
    #     if not filename:
    #         return
    #
    #     absUnit = grazing_units.absolute_unit(dim1_unit)
    #     header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
    #                    "wavelength",
    #                    "sample orientation", "incident angle", "tilt angle",
    #                    "chi_min", "chi_max",
    #                    dim1_unit.REPR + "_min",
    #                    dim1_unit.REPR + "_max",
    #                    "pixelX", "pixelY",
    #                    "dark", "flat", "polarization_factor",
    #                    "normalization_factor"]
    #
    #     header = {"dist": str(self._dist),
    #               "poni1": str(self._poni1),
    #               "poni2": str(self._poni2),
    #               "rot1": str(self._rot1),
    #               "rot2": str(self._rot2),
    #               "rot3": str(self._rot3),
    #               "wavelength" : str(self._wavelength),
    #               "sample orientation" : str(self._sample_orientation),
    #               "incident angle" : str(self._incident_angle),
    #               "tilt angle" : str(self._tilt_angle),
    #               "chi_min": str(dim2.min()),
    #               "chi_max": str(dim2.max()),
    #               dim1_unit.REPR + "_min": str(dim1.min()),
    #               dim1_unit.REPR + "_max": str(dim1.max()),
    #               "pixelX": str(self.pixel2),
    #               "pixelY": str(self.pixel1),
    #               "polarization_factor": str(polarization_factor),
    #               "normalization_factor":str(normalization_factor)
    #               }
    #
    #     if self.splineFile:
    #         header["spline"] = str(self.splineFile)
    #
    #     if dark is not None:
    #         if self.darkfiles:
    #             header["dark"] = self.darkfiles
    #         else:
    #             header["dark"] = 'unknown dark applied'
    #     if flat is not None:
    #         if self.flatfiles:
    #             header["flat"] = self.flatfiles
    #         else:
    #             header["flat"] = 'unknown flat applied'
    #     f2d = self.getFit2D()
    #     for key in f2d:
    #         header["key"] = f2d[key]
    #     try:
    #         img = fabio.edfimage.edfimage(data=I.astype("float32"),
    #                                       header=header,
    #                                       header_keys=header_keys)
    #
    #         if error is not None:
    #             img.appendFrame(data=error,
    #                             header={"EDF_DataBlockID": "1.Image.Error"})
    #         img.write(filename)
    #     except IOError:
    #         logger.error("IOError while writing %s" % filename)
    #
    # def save_transformed(self, filename, I, dim1, dim2,
    #                      error=None, unit=grazing_units.Q,
    #                      dark=None, flat=None, polarization_factor=None,
    #                      normalization_factor=None):
    #     """
    #     This method saves the result of grazing-incidence transformation.
    #     If no filename is given the function will exit.
    #
    #     Parameters
    #     ----------
    #     filename : str
    #         The file name to save the transformed image.
    #     I : ndarray
    #         Transformed image (intensity).
    #     dim1 : ndarray
    #         In-plane coordinates.
    #     dim2 : ndarray
    #         Out-of-plane coordinates.
    #     error : ndarray or None
    #         The error bar for each intensity.
    #     unit : PyGIX.grazing_units.Unit
    #         The in-plane and out-of-plane units.
    #     dark :  ???
    #         Save the dark filename(s) (default: no).
    #     flat : ???
    #         Save the flat filename(s) (default: no).
    #     polarization_factor : float
    #         The polarization factor.
    #     normalization_factor: float
    #         The normalization factor..
    #     """
    #     if not filename:
    #         return
    #
    #     giUnit = grazing_units.to_unit(unit)
    #     ipl_giUnit = grazing_units.ip_unit(giUnit)
    #     opl_giUnit = grazing_units.op_unit(giUnit)
    #     header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
    #                    "sample orientation", "incident angle", "tilt angle",
    #                    ipl_giUnit.IPL_UNIT + "_min",
    #                    ipl_giUnit.IPL_UNIT + "_max",
    #                    opl_giUnit.OPL_UNIT + "_min",
    #                    opl_giUnit.OPL_UNIT + "_max",
    #                    "pixelX", "pixelY",
    #                    "dark", "flat", "polarization_factor",
    #                    "normalization_factor"]
    #
    #     header = {"dist": str(self._dist),
    #               "poni1": str(self._poni1),
    #               "poni2": str(self._poni2),
    #               "rot1": str(self._rot1),
    #               "rot2": str(self._rot2),
    #               "rot3": str(self._rot3),
    #               "sample orientation" : str(self._sample_orientation),
    #               "incident angle" : str(self._incident_angle),
    #               "tilt angle" : str(self._tilt_angle),
    #               ipl_giUnit.IPL_UNIT + "_min": str(dim1.min()),
    #               ipl_giUnit.IPL_UNIT + "_max": str(dim1.max()),
    #               opl_giUnit.OPL_UNIT + "_min": str(dim2.min()),
    #               opl_giUnit.OPL_UNIT + "_max": str(dim2.max()),
    #               "pixelX": str(self.pixel2),  # this is not a bug ... most people expect dim1 to be X
    #               "pixelY": str(self.pixel1),  # this is not a bug ... most people expect dim2 to be Y
    #               "polarization_factor": str(polarization_factor),
    #               "normalization_factor":str(normalization_factor)
    #               }
    #
    #     if self.splineFile:
    #         header["spline"] = str(self.splineFile)
    #
    #     if dark is not None:
    #         if self.darkfiles:
    #             header["dark"] = self.darkfiles
    #         else:
    #             header["dark"] = 'unknown dark applied'
    #     if flat is not None:
    #         if self.flatfiles:
    #             header["flat"] = self.flatfiles
    #         else:
    #             header["flat"] = 'unknown flat applied'
    #     f2d = self.getFit2D()
    #     for key in f2d:
    #         header[key] = f2d[key]
    #     try:
    #         img = fabio.edfimage.edfimage(data=I.astype("float32"),
    #                                       header=header,
    #                                       header_keys=header_keys)
    #
    #         if error is not None:
    #             img.appendFrame(data=error,
    #                             header={"EDF_DataBlockID": "1.Image.Error"})
    #         img.write(filename)
    #     except IOError:
    #         logger.error("IOError while writing %s" % filename)

    def save1D(self, filename, x_scale, intensity, error=None,
               x_unit=grazing_units.Q, has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None):
        """
        Save the data for 1D integration.

        Args:
            filename (string):
            x_scale (ndarray):
            intensity (ndarray):
            error (ndarray or None):
            x_unit (pygix.unit.Unit):
            has_dark (bool):
            has_flat (bool):
            polarization_factor (float or None):
            normalization_factor (float or None):
        """
        # x_unit = grazing_units.to_unit(x_unit)
        with open(filename, "w") as f:
            f.write(self.make_header(
                has_dark=has_dark, has_flat=has_flat,
                polarization_factor=polarization_factor,
                normalization_factor=normalization_factor))
            try:
                f.write("\n# --> %s\n" % filename)
            except UnicodeError:
                f.write("\n# --> %s\n" % (filename.encode("utf8")))
            if error is None:
                f.write("#%14s %14s\n" % (x_unit.REPR, "I "))
                f.write("\n".join(["%14.6e  %14.6e" % (t, i) for t, i in
                                   zip(x_scale, intensity)]))
            else:
                f.write("#%14s  %14s  %14s\n" %
                        (x_unit.REPR, "I ", "sigma "))
                f.write("\n".join(
                    ["%14.6e  %14.6e %14.6e" % (t, i, s) for t, i, s in
                     zip(x_scale, intensity, error)]))
            f.write("\n")


def make_roi_header(**param):
    """
    Format header data to be written when saving ROI data.

    Args:
        method (string): integration method.
        param (dict): integration parameters.

    Returns:
        hdr_list (string): header data.
    """
    hdr_list = ['=== Integration ROI ===']
    method = [i for i in param.keys() if "pos" in i][0].split('_pos')[0]
    hdr_list.append('Integration method: {}'.format(method))

    for k, v in param.items():
        hdr_list.append('{}: {}'.format(k, v))

    header = "\n".join(['# ' + i for i in hdr_list])
    return header


def save_roi(x_data, y_data, filename, **param):
    """


    Args:
        x_data (ndarray): x array of ROI.
        y_data (ndarray): y array of ROI.
        filename (string): path of file to be saved.
        **param (dict): integration parameters to be written the header.

    Returns:

    """
    header = make_roi_header(**param)

    with open(filename, 'w') as f:
        f.write(header)
        f.write('#%14s %14s\n' % ('x', 'y'))
        f.write('\n'.join(
            ['%14.6e  %14.6e' % (t, i) for t, i in zip(x_data, y_data)]))
        f.write('\n')
