#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A collection of useful tools for GIXS and fibre diffraction.

There are two main tools:

1. Four quadrant averaging:
    Fibre diffraction patterns are related through four quadrant symmetry
    (assuming ideal fibre texture). This means that the data in each of the
    four quadrants are equivalent.[1] This allows us to cut out each quadrant,
    centre, rotate and overlay, then average the sum of all four. This has two
    advantages. Firstly, statistics are improved. Secondly, regions of missing
    data (detector module gaps, asymmetric splitting due to large incident
    angle) are filled from other quadrants - prettier images!

    Example usage:
        averaged, x, y = quadrant_average(data, x, y)

    N.B. The span of the image will change and will be determined by
    -absmax(q)) to +absmax(q) such that new x and y scales are returned.

2. Integration ROIs:
    Arrays defining the regions-of-interest (ROIs) for 1D data reduction methods
    can be calculated. These are useful for overlaying on diffraction patterns
    in figures to show the regions of data that have been integrated. Four
    methods exist for each of the four 1D reduction methods:
        sector_roi
        chi_roi
        op_box_roi
        ip_box_roi

    These functions take the same parameters as the integration methods.

    N.B. These are all calculated in the output coordinate space, so must be
    overlayed onto transformed images, NOT raw images.

References:
    [1] Stribeck and Nöchel, J. Appl. Crystallogr., (2009), 42, 295.
"""

import numpy as np
from . import io


def quadrant_average(data, x=None, y=None, dummy=0, filename=None):
    """
    Function to perform four quadrant averaging of fiber diffraction
    patterns. If only the data array is given, the function 
    assumes the center of reciprocal space is the central pixel
    and the returned averaged array will have the same shape as 
    the input data. In general this should be avoided, it is safest if
    x and y scales are provided!

    If x and y scaling are given, it deduces the
    center from these arrays. One quadrant will have the size of 
    the largest quadrant and the resulting array will be larger 
    than the input array. In this case a new x and y will be
    calculated and returned.
    
    Args:
        data (ndarray): Input reciprocal space map array.
        x (ndarray): x scaling of the image.
        y (ndarray: y scaling of the image.
        dummy (int): Value of masked invalid regions.
        filename (string): Name of file to be saved.

    Returns:
        out_full (ndarray): The four quadrant averaged array
    """
    if (x is not None) and (y is not None):
        cen_x = np.argmin(abs(x))
        cen_y = np.argmin(abs(y))
    elif (x is None) and (y is None):
        cen_y = data.shape[0] / 2.0
        cen_x = data.shape[1] / 2.0
    else:
        raise RuntimeError('Must pass both x and y scales or neither')

    quad1 = np.flipud(np.fliplr(data[0:cen_y, 0:cen_x]))
    quad2 = np.fliplr(data[cen_y:, 0:cen_x])
    quad3 = data[cen_y:, cen_x:]
    quad4 = np.flipud(data[0:cen_y, cen_x:])

    quad_shape_y = max(data.shape[0] - cen_y,
                       data.shape[0] - (data.shape[0] - cen_y))
    quad_shape_x = max(data.shape[1] - cen_x,
                       data.shape[1] - (data.shape[1] - cen_x))
    mask = np.zeros((quad_shape_y, quad_shape_x))
    out = np.zeros((quad_shape_y, quad_shape_x))

    out[np.where(quad1 > dummy)] += quad1[np.where(quad1 > dummy)]
    out[np.where(quad2 > dummy)] += quad2[np.where(quad2 > dummy)]
    out[np.where(quad3 > dummy)] += quad3[np.where(quad3 > dummy)]
    out[np.where(quad4 > dummy)] += quad4[np.where(quad4 > dummy)]

    mask[np.where(quad1 > dummy)] += 1
    mask[np.where(quad2 > dummy)] += 1
    mask[np.where(quad3 > dummy)] += 1
    mask[np.where(quad4 > dummy)] += 1

    out[np.where(mask > 0)] /= mask[np.where(mask > 0)]
    out[np.where(mask == 0)] = dummy

    out_full = np.zeros((out.shape[0] * 2, out.shape[1] * 2))
    cen_x = out_full.shape[1] / 2.0
    cen_y = out_full.shape[0] / 2.0

    out_full[0:cen_y, 0:cen_x] = np.flipud(np.fliplr(out))
    out_full[0:cen_y, cen_x:] = np.flipud(out)
    out_full[cen_y:, cen_x:] = out
    out_full[cen_y:, 0:cen_x] = np.fliplr(out)

    if (x is not None) and (y is not None):
        out_x = np.linspace(-abs(x).max(), abs(x).max(), out_full.shape[1])
        out_y = np.linspace(-abs(y).max(), abs(y).max(), out_full.shape[0])
    else:
        out_x = None
        out_y = None

    if filename is not None:
        writer = io.Writer(None, None)
        writer.save2D(filename, out_full, out_x, out_y)

    if (out_x is not None) and (out_y is not None):
        return out_full, out_x, out_y
    else:
        return out_full


def sector_roi(chi_pos=None, chi_width=None, radial_range=None, filename=None):
    """Generate array defining region of interest for sector integration.

    Args:
        chi_pos (float): chi angle (deg) defining the centre of the sector.
        chi_width (float): width (deg) of sector.
        radial_range (tuple): integration range (min, max).
        filename (string): filename to save the arrays.

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    param = locals()  # passed only to io to write key, val in the header
    param.pop('filename')

    if (len([x for x in [chi_pos, chi_width, radial_range] if
             x is not None]) is 0) \
            or (radial_range is None):
        raise RuntimeError('Integration over whole image, no ROI to display.')
    roi_x, roi_y = _calc_sector(radial_range, chi_pos, chi_width)

    if filename:
        io.save_roi(roi_x, roi_y, filename, **param)
    return roi_x, roi_y


def chi_roi(radial_pos, radial_width, chi_range=None, filename=None):
    """Generate array defining region of interest for chi integration.

    Args:
        radial_pos (float): position defining the radius of the sector.
        radial_width (float): width (q or 2th) of sector.
        chi_range (tuple): azimuthal range (min, max).
        filename (string): filename to save the arrays.

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    param = locals()  # passed only to io to write key, val in the header
    param.pop('filename')

    if (chi_range is None) or (chi_range[0] + chi_range[1] is 360):
        chi_width = None
        chi_pos = None
    else:
        chi_width = chi_range[1] - chi_range[0]
        chi_pos = chi_range[0] + chi_width / 2.0

    radial_min = radial_pos - radial_width / 2.0
    radial_max = radial_pos + radial_width / 2.0
    roi_x, roi_y = _calc_sector((radial_min, radial_max), chi_pos, chi_width)

    if filename:
        io.save_roi(roi_x, roi_y, filename, **param)
    return roi_x, roi_y


def op_box_roi(ip_pos, ip_width, op_range, filename=None):
    """Generate array defining region of interest for out-of-plane box integration.

    Args:
        ip_pos (float): in-plane centre of integration box.
        ip_width (float): in-plane width of integration box.
        op_range (tuple): out-of-plane range (min, max).
        filename (string): filename to save the arrays.

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    param = locals()  # passed only to io to write key, val in the header
    param.pop('filename')

    ip_min = ip_pos - ip_width / 2.0
    ip_max = ip_pos + ip_width / 2.0
    roi_x, roi_y = _calc_box((ip_min, ip_max), op_range)

    if filename:
        io.save_roi(roi_x, roi_y, filename, **param)
    return roi_x, roi_y


def ip_box_roi(op_pos, op_width, ip_range, filename=None):
    """Generate array defining region of interest for in-plane box integration.

    Args:
        op_pos (float): out-of-plane centre of integration box.
        op_width (float): out-of-plane width of integration box.
        ip_range (tuple): in-plane range (min, max).
        filename (string): filename to save the arrays.

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    param = locals()  # passed only to io to write key, val in the header
    param.pop('filename')

    op_min = op_pos - op_width / 2.0
    op_max = op_pos + op_width / 2.0
    roi_x, roi_y = _calc_box(ip_range, (op_min, op_max))

    if filename:
        io.save_roi(roi_x, roi_y, filename, **param)
    return roi_x, roi_y


def _calc_sector(radial_range, chi_pos, chi_width):
    """Main function for calculating sector region of interest.
    Called by sector_roi and chi_roi.

    Args:
        radial_range (tuple): integration range (min, max).
        chi_pos (float): chi angle (deg) defining the centre of the sector.
        chi_width (float): width (deg) of sector.

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    if len([x for x in [chi_pos, chi_width] if x is not None]) is 1:
        raise RuntimeError('both chi_pos and chi_width must be supplied or '
                           'neither')

    if (chi_pos is None) and (chi_width is None):
        chi_min = 0
        chi_max = 359
        npts = 360
    else:
        chi_min = -(chi_pos - chi_width / 2.0 - 90.0)
        chi_max = -(chi_pos + chi_width / 2.0 - 90.0)
        npts = abs(int(chi_max - chi_min))

    chi = np.radians(np.linspace(chi_min, chi_max, npts))

    # lower part of arc
    if radial_range[0] is 0:
        lo_qr = np.array(0)
        lo_qz = np.array(0)
    else:
        lo_qr = radial_range[0] * np.cos(chi)
        lo_qz = radial_range[0] * np.sin(chi)

    # upper part of arc
    hi_qr = (radial_range[1] * np.cos(chi))[::-1]
    hi_qz = (radial_range[1] * np.sin(chi))[::-1]

    qr = np.hstack((lo_qr, hi_qr))
    qz = np.hstack((lo_qz, hi_qz))

    if (chi_pos is not None) and (chi_width is not None):
        qr = np.append(qr, qr[0])
        qz = np.append(qz, qz[0])
    return qr, qz


def _calc_box(ip_range, op_range):
    """Main function for calculating box regions of interest.
    Called by op_box_roi and ip_box_roi.

    Args:
         ip_range (tuple): in-plane (min, max).
        op_range (tuple): out-of-plane (min, max).

    Returns:
        qr, qz (tuple of ndarrays): arrays defining the region of interest.
    """
    qr = [ip_range[0], ip_range[0], ip_range[1], ip_range[1], ip_range[0]]
    qz = [op_range[0], op_range[1], op_range[1], op_range[0], op_range[0]]
    return qr, qz
