#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" input/output functions.
"""

from collections import OrderedDict
import fabio
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
            hdr_dict = OrderedDict()

            if self._pg is None:
                self._hdr_dict = hdr_dict
                return self._hdr_dict

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

    def make_header_string(self, header_dict=None, hdr_key='#',
                           has_dark=False, has_flat=False,
                           polarization_factor=None, normalization_factor=None):
        """

        Args:
            header_dict:
            hdr_key:
            has_dark:
            has_flat:
            polarization_factor:
            normalization_factor:

        Returns:

        """
        if (header_dict is None) and (self._hdr_dict is None):
            self.make_header_dict(has_dark, has_flat, polarization_factor,
                                  normalization_factor)
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
            hdr_key (string): string used as comment in the header.
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
                self._header = self.make_header_string(hdr_key=hdr_key)
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

    def save2D(self, filename, intensity, x_scale=None, y_scale=None,
               error=None, hdr_note=None, has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None):
        """

        Args:
            filename:
            intensity:
            x_scale:
            y_scale:
            error:
            hdr_note (string):
            has_dark:
            has_flat:
            polarization_factor:
            normalization_factor:

        Returns:

        """
        if self._hdr_dict is None:
            self.make_header_dict(has_dark, has_flat, polarization_factor,
                                  normalization_factor)
        header = self._hdr_dict
        if hdr_note is not None:
            header['Note'] = hdr_note
        if (x_scale is not None) and (y_scale is not None):
            header['== data scaling =='] = ''
            header['x min'] = x_scale[0]
            header['x max'] = x_scale[-1]
            header['y min'] = y_scale[0]
            header['y max'] = y_scale[-1]

        try:
            img = fabio.edfimage.edfimage(data=intensity.astype("float32"),
                                          header=header)
            if error is not None:
                img.appendFrame(data=error,
                                header={"EDF_DataBlockID": "1.Image.Error"})
            img.write(filename)
        except IOError:
            print("IOError while writing %s" % filename)


def make_roi_header(**param):
    """
    Format header data to be written when saving ROI data.

    Args:
        method (string): integration method.
        param (dict): integration parameters.

    Returns:
        hdr_list (string): header data.
    """
    hdr_list = ['== Integration ROI ==']
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
        f.write('\n#%14s %14s\n' % ('x', 'y'))
        f.write('\n'.join(
            ['%14.6e  %14.6e' % (t, i) for t, i in zip(x_data, y_data)]))
        f.write('\n')
