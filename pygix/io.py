#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" input/output functions.
"""

import fabio
from . import grazing_units


class Writer(object):
    """

    """

    def __init__(self, pg, filename):
        """

        Returns:

        """
        print pg
        self._pg = pg
        self._filename = filename
        self._hdr_dict = None
        self._header = None
        self._already_written = False

    def make_header_dict(self, has_dark=False, has_flat=False,
                         polarization_factor=None, normalization_factor=None):
        """

        Args:
            hdr (string): string used as comment in the header.
            has_dark (bool): save the file names of dark images.
            has_flat (bool): save the file names of flat images.
            polarization_factor (float):
            normalization_factor (float):

        Returns:
            self._hdr_dict (OrderedDict):
        """
        if self._hdr_dict is None:
            try:
                from collections import OrderedDict
                hdr_dict = OrderedDict()
            except ImportError:
                hdr_dict = {}

            pg = self._pg
            hdr_dict['== pyFAI calibration =='] = ''
            hdr_dict['Spline file'] = pg.splineFile
            hdr_dict['Pixel size (m)'] = (pg.pixel1, pg.pixel2)
            hdr_dict['PONI (m)'] = (pg.poni1, pg.poni2)
            hdr_dict['Sample-detector distance (m)'] = pg.dist
            hdr_dict['Rotations (rad)'] = (pg.rot1, pg.rot2, pg.rot3)
            hdr_dict['wavelength'] = pg.wavelength

            hdr_dict['== pygix parameters =='] = ''
            hdr_dict['Sample orientation'] = pg.sample_orientation
            hdr_dict['Incident angle (degrees)'] = pg.incident_angle
            hdr_dict['Tilt angle (degrees)'] = pg.tilt_angle

            hdr_dict['== corrections =='] = ''
            if pg.maskfile is not None:
                hdr_dict['Mask file'] = pg.maskfile
            if has_dark or (pg.darkcurrent is not None):
                if pg.darkfiles:
                    hdr_dict['Dark current'] = pg.darkfiles
                else:
                    hdr_dict['Dark current'] = 'Unknown dark file'
            if has_flat or (pg.flatfield is not None):
                if pg.flatfiles:
                    hdr_dict['Flat field'] = pg.flatfiles
                else:
                    hdr_dict['Flat field'] = 'Unknown flat file'
            if (polarization_factor is None) and (pg._polarization is not None):
                polarization_factor = pg._polarization_factor
            hdr_dict['Polarization factor'] = polarization_factor
            hdr_dict['Normalization factor'] = normalization_factor
            self._hdr_dict = hdr_dict
        return self._hdr_dict

    def header_dict_to_string(self, header_dict=None, hdr_key='#'):
        """

        Args:
            header_dict:

        Returns:

        """
        if header_dict is None:
            header_dict = self._hdr_dict

        hdr_list = []
        for k, v in header_dict.items():
            if (v.__class__ is str) and (len(v) is 0):
                hdr_list.append(k)
            else:
                hdr_list.append('{}: {}'.format(k, v))
        return '\n'.join(['{} {}'.format(hdr_key, i) for i in hdr_list])

    def make_header(self, hdr_key="#", has_dark=False, has_flat=False,
                    polarization_factor=None, normalization_factor=None):
        """

        Args:
            hdr (string): string used as comment in the header.
            has_dark (bool): save the file names of dark images.
            has_flat (bool): save the file names of flat images.
            polarization_factor (float):
            normalization_factor (float):

        Returns:

        """
        if self._header is None:
            if self._hdr_dict is None:
                self.make_header_dict(has_dark, has_flat, polarization_factor,
                                      normalization_factor)
                self._header = self.header_dict_to_string(hdr_key=hdr_key)
        return self._header

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
            
    def save2D(self, filename, intensity, x_scale, y_scale, error=None,
               has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None):
        """
        """
        if self._hdr_dict is None:
            self.make_header_dict(has_dark, has_flat, polarization_factor,
                                  normalization_factor)

        header = self._hdr_dict
        header['== data scaling =='] = ''
        header['x min'] = x_scale[0]
        header['x max'] = x_scale[-1]
        header['y min'] = y_scale[0]
        header['y max'] = y_scale[-1]

        try:
            img = fabio.edfimage.edfimage(data=intensity.astype("float32"),
                                          header=header)

            if error is not None:
                img.appendFrame(data=error, header={"EDF_DataBlockID": "1.Image.Error"})
            img.write(filename)
        except IOError:
            print "IOError while writing %s" % filename


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
