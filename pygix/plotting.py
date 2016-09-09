#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl_version = [i for i in mpl.__version__.split('.')]
if int(mpl_version[0]) == 1:
    if int(mpl_version[1]) >= 5:
        default_cmap = 'inferno'
    else:
        default_cmap = 'afmhot'
else:
    default_cmap = 'hot'
colormap = getattr(plt.cm, default_cmap)
lcolor = colormap(95)

# Settings for plotting
font_dict = {'family': 'sans-serif',
             'style': 'normal',
             'weight': 'normal',
             'sans-serif': 'Helvetica',
             'size': 9}

fig_dict = {'figsize': (3.25, 3.25),
            'dpi': 140}

axes_dict = {'linewidth': 0.75,
             'labelpad': 8}

lines_dict = {'linewidth': 0.75,
              'markersize': 3}

mpl.rc('font', **font_dict)
mpl.rc('figure', **fig_dict)
mpl.rc('axes', **axes_dict)
mpl.rc('lines', **lines_dict)
mpl.rc('text', usetex=True)

latexpre = [r'\usepackage{siunitx}',
            r'\sisetup{detect-all}',
            r'\usepackage{helvet}',
            r'\usepackage[EULERGREEK]{sansmath}',
            r'\sansmath']
mpl.rcParams['text.latex.preamble'] = latexpre

DEFAULT_UNIT = 'nm^-1'
LABELS_DICT = {
    'raw': ['y (pixels)', 'x (pixels)'],
    'angular': ['2theta_f (deg)', 'alpha_f (deg)'],
    'rsm': ['q_xy (nm^-1)', 'q_z (nm^-1)'],
    'polar': ['q (nm^-1)', 'chi (deg)'],
    'q': ['q (nm^-1)', 'Intensity (a.u.)'],
    'chi': ['chi (deg)', 'Intensity (a.u.)'],
    'qz': ['q_z (nm^-1)', 'Intensity (a.u.)'],
    'qxy': ['q_xy (nm^-1)', 'Intensity (a.u.)']}


def show():
    """Wrapper for plt.show() to save having to separately import mpl"""
    plt.show()


def get_line_colors(cmap, ncolors):
    """
    Automatically generate a list of rgb colors for line colors in a 1D
    plot.

    Args:
        cmap: (string) name of an mpl color map (or cmap instance).
        ncolors: number of colors to be returned in the list.

    Returns:
        lcolors: a list of rgb tuples.
    """
    cmap = getattr(plt.cm, cmap)
    stop = 0.8 * 255
    lcolors = [cmap(int(i)) for i in np.linspace(0, stop, ncolors)]
    return lcolors


def get_axis_label(label=None):
    """

    Args:
        label:

    Returns:

    """
    if (label is None) or (len(label) is 0):
        return ''
    if '(' in label:
        quantity, unit = label.split('(')
        quantity = quantity.strip(' ')
        unit = unit.strip(')')
    else:
        quantity = label
        unit = None

    if not unit and quantity.startswith('q'):
        unit = DEFAULT_UNIT

    if '_' in quantity:
        quantity, quant_sub = quantity.split('_')
    else:
        quant_sub = None

    if unit and ('^' in unit):
        unit, unit_sup = unit.split('^')
    else:
        unit_sup = None

    if unit == 'um':
        unit = '\si{\micro\meter}'

    axis_label = r''
    if quantity.startswith('2'):
        axis_label += r'2'
        quantity = quantity.strip('2')
    if (quantity.lower() in ['time', 'distance', 'intensity']) or (
            'point' in quantity.lower()):
        quantity = quantity.replace(" ", "\ ")
        axis_label += '$\mathrm{%s}' % quantity
    elif quantity in ['alpha', 'beta', 'gamma', 'chi', 'psi', 'theta']:
        axis_label += r'$\%s' % quantity
    else:
        axis_label += r'$\textit{%s}' % quantity
    if quant_sub:
        axis_label += r'_{%s}' % quant_sub
    axis_label += r'$'
    if unit:
        if unit == 'deg':
            axis_label += ' ($^{\circ}$)'
        else:
            axis_label += ' ($\mathrm{%s}' % unit
            if unit_sup:
                axis_label += '^{%s}' % unit_sup
            axis_label += '$)'
    return axis_label


def plot_command(x, y, logx=False, logy=False, **kwargs):
    """
    Args:
        x:
        y:
        logx:
        logy:
        **kwargs:

    Returns:

    """
    if logy and not logx:
        line, = plt.semilogy(x, y, **kwargs)
    elif logx and not logy:
        line, = plt.semilogx(x, y, **kwargs)
    elif logx and logy:
        line, = plt.loglog(x, y, basex=10, **kwargs)
        # raise NotImplementedError('logx and logy not implemented')
    else:
        line, = plt.plot(x, y, **kwargs)
    return line


def plot(x, y,
         xlim=None, ylim=None,
         mode='q', xlabel=None, ylabel=None,
         logx=False, logy=False,
         legend=None, line_cmap=None,
         tight_layout=True, filename=None,
         show=True, newfig=True):
    """

    Args:
        x:
        y:
        xlim:
        ylim:
        labelx:
        labely:
        logx:
        logy:
        legend:
        line_cmap:
        tight_layout:
        filename:
        show:
        newfig:

    Returns:

    """
    if newfig:
        fig = plt.figure()

    kwargs = {}
    if y.__class__ == np.ndarray:
        if x.__class__ != np.ndarray:
            raise RuntimeError(('pygix.plotting.plot: y data is ndarray, x data'
                                ' is', x.__class__))
        kwargs['color'] = lcolor
        plot_command(x, y, logx, logy, **kwargs)
        xlim = (x[0], x[-1])
    elif y.__class__ == list:
        lines = []
        if line_cmap is None:
            line_cmap = default_cmap
        line_colors = get_line_colors(line_cmap, len(y))

        if x.__class__ == np.ndarray:
            for i, yy in enumerate(y):
                kwargs['color'] = line_colors[i]
                lines.append(plot_command(x, yy, logx, logy, **kwargs))
        elif x.__class__ == list:
            if len(x) is not len(y):
                raise RuntimeError('pygix.plotting.plot: x and y are lists but '
                                   'do not have the same length')
            for xx, yy in zip(x, y):
                lines.append(plot_command(xx, yy, logx, logy, **kwargs))
    else:
        raise RuntimeError('pygix.plotting.plot: y data must be numpy.ndarray'
                           ' or list of numpy.ndarrays')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if mode:
        try:
            tmp_xlabel, tmp_ylabel = LABELS_DICT[mode]
        except KeyError:
            raise RuntimeError('Invalid mode for pygix.plotting.plot\n')
        if xlabel is None:
            xlabel = tmp_xlabel
        if ylabel is None:
            ylabel = tmp_ylabel
    plt.xlabel(get_axis_label(xlabel))
    plt.ylabel(get_axis_label(ylabel))

    # set the apsect ratio to square, problem is, if you use the
    # zoom function on the widget the data apsect remains the same
    # ax = plt.gca()
    # x0, x1 = ax.get_xlim()
    # y0, y1 = ax.get_ylim()
    # ax.set_aspect((x1-x0)/(y1-y0))

    # need to handle position with bbox_to_anchor
    if legend:
        if len(legend) != len(lines):
            raise RuntimeError('pygix.plotting.plot: not enough legend labels'
                               ' for number of lines')
        plt.legend(lines, legend)

    if tight_layout:
        plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()


# def rsmplot(data, x=None, y=None, xlim=None, ylim=None,
#             xlabel=None, ylabel=None, cmap=None, colorbar=False, clim='auto',
#             newfig=True, show=True, figsize=(7, 7), tight_layout=True,
#             filename=None, **kwargs):


def implot(data, x=None, y=None, mode=None,
           xlim=None, ylim=None, xlabel=None, ylabel=None,
           cmap=None, clim='auto', colorbar=False,
           newfig=True, show=True, tight_layout=True,
           filename=None,
           **kwargs):
    """
       Parameters
    ----------
    data : ndarray
        Image data array
    x : ndarray
        x axis scale
    y : ndarray
        y axis scale
    mode : string
        Data type. Accepted values: raw, angular, rsm
    xlim : tuple
        (lower, upper) limit of x axis
    ylim : tuple
        (lower, upper) limit of y axis
    cmap : string
        Valid name of matplotlib colormap
    colorbar : bool
        Add colorbar or not
    clim : 'auto', None or tuple
        z-scaling of image data. If 'auto' will set based on
        (0.5, 99.5) percentile of intensity values; if None uses
        (data.min(), data.max()); otherwise tuple of (min, max).
    newfig : bool
        Specify whether to generate a new figure or not. This is
        useful if you are creating a figure with many subplots,
        using newfig=False will not make a new figure window.
    show : bool
        Specify whether to call plt.show() when plot has been
        created. Again, useful for subplots or batch processing.
    filename : string or None
        If a string is passed the plot will be saved with given
        filename.

    Returns
    -------
    fig : matplotlib.pyplot.figure instance
        If newfig=True, returns the figure object.
    im : matplotlib.pyplot.imshow instance
        Returns only im object if newfig=False, otherwise
        (fig, im).
    """
    if mode and (mode not in ['raw', 'angular', 'rsm', 'polar']):
        raise RuntimeError('pygix.plotting.implot: mode must be raw, angular'
                           ' or rsm')
    if mode == 'raw':
        origin = 'upper'
    else:
        origin = 'lower'

    try:
        xmin, xmax = x[0], x[-1]
    except TypeError:
        xmin, xmax = 0, data.shape[1]
    try:
        ymin, ymax = y[0], y[-1]
    except TypeError:
        ymin, ymax = 0, data.shape[0]
    extent = [xmin, xmax, ymin, ymax]

    if not cmap:
        cmap = default_cmap

    if clim is None:
        vmin, vmax = data.min(), data.max()
    elif clim == 'auto':
        vmin, vmax = np.percentile(data, (0.1, 99.5))
    else:
        vmin, vmax = clim

    if newfig:
        fig = plt.figure()

    if mode == 'polar':
        if kwargs is None:
            kwargs = {}
        kwargs['aspect'] = 0.2

    im = plt.imshow(data,
                    origin=origin,
                    vmin=vmin, vmax=vmax,
                    interpolation='nearest',
                    cmap=cmap,
                    extent=extent,
                    **kwargs)
    ax = plt.gca()

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if mode:
        try:
            tmp_xlabel, tmp_ylabel = LABELS_DICT[mode]
        except KeyError:
            raise RuntimeError('Invalid mode for pygix.plotting.implot\n')
        if xlabel is None:
            xlabel = tmp_xlabel
        if ylabel is None:
            ylabel = tmp_ylabel
    plt.xlabel(get_axis_label(xlabel))
    plt.ylabel(get_axis_label(ylabel))

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    ax.set_axis_bgcolor(colormap(0))

    if tight_layout:
        plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    if newfig:
        return fig, im, ax
    else:
        return im, ax
