import sys
import os
from glob import glob
import yaml
import fabio
import pygix
import numpy as np
import time

import matplotlib.pyplot as plt


class Processor(object):
    """
    Class for batch processing data with pygix module. Takes a
    yaml file, which lists geometry, correction files, input data,
    output file names and integration parameters and performs the
    data reduction.
    """

    def __init__(self, yaml_file):
        """
        Initialization. Takes the yaml_file and reads in all 
        parameters, which are stored as class attributes. Instatiates
        pygix.transform.Transform integrator.
        """
        self.recipe = yaml_file
        with open(yaml_file, 'r') as f:
            self.pars = yaml.load(f)

        try:
            self.live_view = self.pars['live_view']
        except KeyError:
            self.live_view = False
        try:
            self.interactive = self.pars['interactive']
        except KeyError:
            self.interactive = False

        try:
            calibration = self.pars['calibration']
        except KeyError:
            raise RuntimeError('calibration data not present in yaml file')
        self._gi = self.init_pygix(calibration)

        try:
            data = self.pars['data']
        except KeyError:
            raise RuntimeError('data information not present in yaml file')
        self.file_list = self.get_file_list(data['infiles']['dname'],
                                            data['infiles']['fname'],
                                            data['infiles']['numbers'])

        if 'outfiles' in data.keys():
            raise NotImplementedError('Saving data not yet implemented')

        # self.bkg_dname = data['backfiles']['dname']
        # self.bkg_fname = data['backfiles']['fname']
        # self.bkg_numbers = data['backfiles']['numbers']
        # self.out_dname = data['outfiles']['dname']
        # self.out_fname = data['outfiles']['fname']

        # set basename and extn checks for compressed formats like file.edf.bz2
        # basename, extn = os.path.splitext(self.in_fname)
        # compressed_formats = ['bz2', 'gz']
        # if extn in compressed_formats:
        #     self.basename = os.path.splitext(basename)[0]
        #     self.extension = os.path.splitext(basename)[1] + '.' + extn
        # else:
        #     self.basename = basename
        #     self.extension = extn

        red_methods = ['transform_reciprocal', 'transform_polar',
                       'transform_angular', 'profile_sector',
                       'profile_chi', 'profile_op_box', 'profile_ip_box']
        try:
            reduction = self.pars['data_reduction']
        except KeyError:
            raise RuntimeError('reduction parameters not in yaml file')

        self.reduction = reduction.keys()[0]
        if self.reduction not in red_methods:
            raise RuntimeError(('Invalid reduction method: %s \nValid reduction'
                                ' methods: %s') % (self.reduction, red_methods))

        red_kwargs = reduction[self.reduction]

        # set as class attribute and remove if value is None otherwise
        # can interfere with integration if kwarg = None
        self.red_kwargs = {}
        if reduction[self.reduction].__class__ is dict:
            for key in red_kwargs:
                if red_kwargs[key] is not None:
                    self.red_kwargs.update({key: red_kwargs[key]})

            for key in self.red_kwargs:
                if 'range' in key:
                    rng = ''.join(
                        c for c in self.red_kwargs[key] if c not in '()')
                    rng = rng.strip().split(',')
                    self.red_kwargs[key] = tuple(float(x) for x in rng)

        self._reducer = self.set_reducer()

    @staticmethod
    def init_pygix(calibration):
        """
        Instantiate a pygix.trasnform.Transform() class instance.
        All parameters are set from the yaml file.
        :param calibration:
        """
        pg = pygix.transform.Transform()
        pg.load(calibration['ponifile'])

        if 'splinefile' in calibration.keys():
            pg.splinefile = calibration['splinefile']
        if 'flatfile' in calibration.keys():
            pg.flatfiles = calibration['flatfile']
        if 'darkfile' in calibration.keys():
            pg.darkfiles = calibration['darkfile']
        if 'maskfile' in calibration.keys():
            pg.maskfile = calibration['maskfile']

        grazing = calibration['grazing_parameters']
        if 'sample_orientation' in grazing.keys():
            pg.sample_orientation = grazing['sample_orientation']
        else:
            pg.sample_orientation = 1

        pg.incident_angle = grazing['incident_angle']
        if 'tilt_angle' in grazing.keys():
            pg.tilt_angle = grazing['tilt_angle']
        return pg

    def set_reducer(self):
        """
        Returns the reduction function (called in __init__ and set as
        class attribute. 
        """
        return getattr(self._gi, self.reduction)

    @staticmethod
    def indices_to_list(indices):
        """Return an abbreviated string representing indices.
        e.g. indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]
             index_string = '1-3, 5-6, 8-13, 15, 20'
        """
        index_string = ""
        end = start = indices[0]
        for i in range(1, len(indices)):
            if indices[i] == (indices[i - 1] + 1):
                end = indices[i]
            else:
                if start == end:
                    index_string += str(start) + ","
                else:
                    index_string += str(start) + "-" + str(end) + ","
                start = end = indices[i]
        if start == end:
            index_string += str(start)
        else:
            index_string += str(start) + "-" + str(end)
        return index_string

    @staticmethod
    def list_to_indices(index_string):
        """Return an integer list from a string representing indices.
        e.g. index_string = '1-3, 5-6, 8-13, 15, 20'
             indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]
        """
        indices = []
        for s in index_string.split(','):
            if '-' in s:
                first, last = s.split('-')
                for i in range(int(first), int(last) + 1):
                    indices.append(i)
            else:
                indices.append(int(s))
        return indices

    def get_file_list(self, dname, fname, numbers=None):
        """ """
        if not os.path.isdir(dname):
            raise IOError('Directory does not exist!\n{}'.format(dname))
        elif (numbers is None) and ('*' not in fname):
            raise IOError(
                'No filenumbers provided and no wildcard (*) in filename')

        compressed_formats = ['bz2', 'gz']

        if (numbers is None) and ('*' in fname):
            file_list = sorted(glob(os.path.join(dname, fname)))
        else:
            if '*' in fname:
                fname = fname.replace('*', '{:04d}')
            else:
                basename, extn = os.path.splitext(fname)
                if extn in compressed_formats:
                    basename, tmp_extn = os.path.splitext(basename)
                    extn = '{}{}'.format(tmp_extn, extn)

                zero_stripped = basename.rstrip('0')
                nzeros = len(basename) - len(zero_stripped)
                if nzeros > 0:
                    fname = '{}{{:0{}d}}{}'.format(zero_stripped, nzeros, extn)
                elif '{:0' not in fname:
                    raise IOError('bad filename specifier')

            if numbers.__class__ == str:
                numbers = self.list_to_indices(numbers)

            file_list = []
            for i in numbers:
                in_file = fname.format(i)
                file_list.append(os.path.join(dname, in_file))
        return file_list

    def do_reduction(self, filename):
        """
        Loads the data from filename and returns the reduced data.
        """
        data = fabio.open(filename).data
        if os.path.splitext(filename)[1] == '.tif':
            data = np.flipud(data)
        return self._reducer(data, **self.red_kwargs)

    def _process1d(self):
        """
        :return:
        """
        file_list = self.file_list
        n_files = len(file_list)
        out_array = np.zeros((n_files, self.red_kwargs['npt']))

        t_start = time.time()

        i = 0
        if self.live_view:
            first_file = file_list[0]
            print 'processing file: 1/{}\n\t{}'.format(n_files, first_file)

            #    out_fname = self.red_dname+self.red_fname % fnum
            #    self.red_kwargs['filename'] = red_fname
            out_array[i, ], xscale = self.do_reduction(first_file)

            ax, graph = self.init_plot(xscale, out_array[i, ], file_list[0])
            if self.interactive:
                raw_input()
            i = 1

        for i in range(i, len(file_list)):
            in_file = file_list[i]
            print 'processing file: {}/{}\n\t{}'.format(i + 1, n_files, in_file)

            # THIS WORKS BUT NEEDS TESTING
            # if self.out_fname is not None:
            #    out_fname = rootname+str(i).zfill(nz)+'_p'+self.extension
            #    self.red_kwargs['filename']  = os.path.join(self.out_dname, out_fname)

            out_array[i, ], xscale = self.do_reduction(in_file)

            if self.live_view:
                self.update_plot(out, ax, graph, file_list[i])
                if self.interactive:
                    raw_input()

        if not self.live_view:
            t_end = time.time() - t_start
            msg = '{} files processed in {} seconds\n'
            msg += '{} seconds per file'

            print msg.format(n_files, t_end, t_end/n_files)

        yscale = np.arange(0, n_files)
        out = (out_array, xscale, yscale)

        # svarray = fabio.edfimage.edfimage(out_array)
        # svarray.write('test.edf')

        # ax, graph = self.init_plot(out, 'summary')
        # plt.show()
        # raw_input()
        return out

    def _process2d(self):
        """
        :return:
        """
        file_list = self.file_list

        i = 0
        if self.live_view:
            first_file = file_list[0]
            print 'processing file: 1/{}\n\t{}'.format(len(file_list),
                                                       first_file[-80:])

            #    out_fname = self.red_dname+self.red_fname % fnum
            #    self.red_kwargs['filename'] = red_fname
            out = self.do_reduction(first_file)

            ax, graph = self.init_plot(out, file_list[0])
            if self.interactive:
                raw_input()
            i = 1

        for i in range(i, len(file_list)):
            in_file = file_list[i]
            print 'processing file: {}/{}\n\t{}'.format(i + 1,
                                                        len(file_list), in_file)

            # THIS WORKS BUT NEEDS TESTING
            # if self.out_fname is not None:
            #    out_fname = rootname+str(i).zfill(nz)+'_p'+self.extension
            #    self.red_kwargs['filename']  = os.path.join(self.out_dname, out_fname)
            out = self.do_reduction(in_file)

            if self.live_view:
                self.update_plot(out, ax, graph, file_list[i])
                if self.interactive:
                    raw_input()
        return True

    def process(self):
        """
        Main batch processor. Will get file list the iterate over 
        each performing reduction. Will save the data if outfiles 
        have been specified. Will plot all if live_view is True,
        requests user input if interactive is True.
        """
        # CHECKS FOR WRITING DATA
        # if (self.out_fname is not None) and (self.out_dname is None):
        #    raise RuntimeError(('Cannot write reduced data as no reduced '
        #        'directory specified.'))
        # if (self.out_dname is not None) and (self.out_fname is None):
        #    self.out_fname = self.basename.rstrip('0')
        # elif (self.out_fname is not None) and (self.out_dname is not None):
        #    rootname = self.basename.rstrip('0')
        #    nz = len(self.basename)-len(rootname)

        if 'profile' in self.reduction:
            out = self._process1d()
        elif 'transform' in self.reduction:
            out = self._process2d()
        return out

    # --------------------------------------------------------------------------
    # plotting functions

    def get_img_scaling(self, data):
        """ 
        Get the vmin and vmax colour values to be used when plotting
        2D data. Calculated from .9 and 99.1 percentile of pixel
        intensities.
        """
        if 'dummy' in self.red_kwargs:
            dummy = self.red_kwargs['dummy']
        else:
            dummy = data.min()
        ma = np.ma.masked_equal(data, dummy, copy=False)
        ma = np.ma.compressed(ma)
        return np.percentile(ma, (0.5, 99.5))

    def get_xy_labels(self):
        """
        Get plot labels based on reduction method and units.
        """
        qtemp = '$q%s\ (%s^{-1})$'
        atemp = '$2\\theta%s\ (%s)$'

        red = self.reduction
        if 'unit' in self.red_kwargs.keys():
            unit = self.red_kwargs['unit']
        else:
            unit = ''
        if 'theta' in unit:
            temp = atemp
            if 'deg' in unit:
                un = '^{\circ}'
            else:
                un = '$\mathrm{radians}$'
        else:
            temp = qtemp
            if ('nm' in unit) or (len(unit) is 0):
                un = '\mathrm{nm}'
            else:
                un = '\AA'
        if 'profile' in red:
            ylab = '$\mathrm{Intensity\ (a.u.)}$'
            if 'chi' in red:
                xlab = '$\chi\ (^{\circ})$'
            else:
                if 'op' in red:
                    x1 = '_{z}'
                elif 'ip' in red:
                    x1 = '_{xy}'
                else:
                    x1 = ''
                xlab = temp % (x1, un)
        else:
            if 'transform' in red:
                x1 = '_{xy}'
                y1 = '_{z}'

                xlab = temp % ('_{xy}', un)
                ylab = temp % ('_{z}', un)
            else:
                xlab = temp % ('', un)
                ylab = '$\chi\ (^{\circ})$'
        return xlab, ylab

    def init_plot(self, out, title):
        """
        Initiates an active plot window.
        """
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        plt.gcf().subplots_adjust(left=0.15, bottom=0.15)

        if ('profile' in self.reduction) and ('summary' not in title):
            plt.ion()
            y, x = out
            graph, = plt.plot(x, y, '-o')
        else:
            data, x, y = out
            vmin, vmax = self.get_img_scaling(data)
            if self.reduction == 'transform_image':
                aspect = 'auto'
            else:
                aspect = (abs(x[0] - x[-1]) / abs(y[0] - y[-1])) * 0.667

            graph = plt.imshow(data, interpolation='nearest',
                               animated=True, origin='lower', cmap='bone',
                               extent=(x[0], x[-1], y[0], y[-1]),
                               vmin=vmin, vmax=vmax, aspect=aspect)
            plt.colorbar()
            plt.ion()

        xlab, ylab = self.get_xy_labels()
        if 'summary' in title:
            ylab = 'Scan point'
        plt.xlabel(xlab, fontsize=20, labelpad=18)
        plt.ylabel(ylab, fontsize=20, labelpad=18)
        plt.title(title, fontsize=16, y=1.04)
        plt.draw()
        return ax, graph

    def update_plot(self, out, ax, graph, title):
        """
        Updates plot window with new data.
        """
        if 'profile' in self.reduction:
            y, x = out
            graph.set_data(x, y)
            ax.relim()
            ax.autoscale_view(True, True, True)
        else:
            data, x, y = out
            vmin, vmax = self.get_img_scaling(data)
            graph.set_data(data)
            plt.clim(vmin, vmax)
        plt.title(title)
        plt.draw()

        # --------------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print 'usage: process.py recipe.yaml'
        sys.exit(0)
    else:
        rp = Processor(sys.argv[1])
        rp.process()
