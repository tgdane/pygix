#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Processor module for batch processing data reduction.

The user creates a yaml file[1] containing all of the detector and sample
geometry parameters, the input files, output files and the data reduction
parameters. Passing this file to the processor, all input files will be
processed and output data saved if requested. If batch 1D is used, an array
containing the stacked 1D profiles will be returned from the processor.

Usage:

import pygix.process as ppr

processor = ppr.Process('process_recipe.yaml')
out = processor.process()

"""

import sys
import os
from glob import glob
import yaml
import fabio
import pygix
import numpy as np
import time


class Processor(object):
    """
    Class for batch processing data with pygix module. Takes a
    yaml file, which lists geometry, correction files, input data,
    output file names and integration parameters and performs the
    data reduction.
    """

    def __init__(self, yaml_file):
        """
        Initialization. Takes the yaml_file and reads in all parameters, which
        are stored as class attributes. Instatiates pygix.transform.Transform.

        Args:
            yaml_file (string): path to recipe yaml file.
        """
        self.recipe = yaml_file
        with open(yaml_file, 'r') as f:
            self.pars = yaml.load(f)

        try:
            calibration = self.pars['calibration']
        except KeyError:
            raise RuntimeError('calibration data not present in yaml file')
        self._pg = init_pygix(calibration)

        try:
            data = self.pars['data']
        except KeyError:
            raise RuntimeError('data information not present in yaml file')
        self.file_list = get_file_list(data['infiles']['dname'],
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

        red_methods = ['transform_reciprocal',
                       'transform_polar',
                       'transform_angular',
                       'profile_sector',
                       'profile_chi',
                       'profile_op_box',
                       'profile_ip_box']

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

    def set_reducer(self):
        """
        Returns the reduction function (called in __init__ and set as
        class attribute.
        """
        return getattr(self._pg, self.reduction)

    def do_reduction(self, filename):
        """
        Loads the data from filename and returns the reduced data.

        Args:
            filename (string): path to data.

        Returns:
            data (ndarray): reduced data. This can be 1D or 2D depending on
                what was specified in the yaml file.
        """
        data = fabio.open(filename).data
        if os.path.splitext(filename)[1] == '.tif':
            data = np.flipud(data)
        return self._reducer(data, **self.red_kwargs)

    def _process1d(self):
        """ Core function for batch processing data (1D reduction).

        All information is stored as class attributes
        """
        file_list = self.file_list
        n_files = len(file_list)
        out_array = np.zeros((n_files, self.red_kwargs['npt']))

        t_start = time.time()

        for i, f in enumerate(file_list):
            print('processing file: {}/{}\n\t{}'.format(i + 1, n_files, f))

            # THIS WORKS BUT NEEDS TESTING
            # if self.out_fname is not None:
            #    out_fname = rootname+str(i).zfill(nz)+'_p'+self.extension
            #    self.red_kwargs['filename']  = os.path.join(self.out_dname,
            #       out_fname)

            out_array[i, ], x_scale = self.do_reduction(f)

        t_end = time.time() - t_start
        msg = '{} files processed in {} seconds\n'
        msg += '{} seconds per file'
        print(msg.format(n_files, t_end, t_end/n_files))

        y_scale = np.arange(0, n_files)
        out = (out_array, x_scale, y_scale)

        # sv_array = fabio.edfimage.edfimage(out_array)
        # sv_array.write('test.edf')
        return out

    def _process2d(self):
        """
        :return:
        """
        file_list = self.file_list
        n_files = len(file_list)

        t_start = time.time()

        for i, f in enumerate(file_list):
            print('processing file: {}/{}\n\t{}'.format(i + 1, n_files, f))

            # THIS WORKS BUT NEEDS TESTING
            # if self.out_fname is not None:
            #    out_fname = rootname+str(i).zfill(nz)+'_p'+self.extension
            #    self.red_kwargs['filename']  = os.path.join(self.out_dname,
            #       out_fname)
            self.do_reduction(f)

        t_end = time.time() - t_start
        msg = '{} files processed in {} seconds\n'
        msg += '{} seconds per file'
        print(msg.format(n_files, t_end, t_end/n_files))

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


def list_to_indices(index_string):
    """
    Return an integer list from a string representing indices.
    e.g. index_string = '1-3, 5-6, 8-13, 15, 20'
         indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]

    Args:
        index_string (string): condensed string representation of integer list.

    Returns:
        indices (list):
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


def indices_to_list(indices):
    """
    Return an abbreviated string representing indices.
    e.g. indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]
         index_string = '1-3, 5-6, 8-13, 15, 20'

    Args:
        indices (list):

    Returns:
        index_string (string): condensed string representation of integer list.
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


def get_file_list(dname, fname, numbers=None):
    """
    Takes a directory path, filename format and (optionally) frame numbers and
    returns a list of full file paths.

    Args:
        dname (string): directory path.
        fname (string): basename for images. Can be full filename or can contain
            wildcard ('*', or '0000').
        numbers: (string, list or None): contains information on images with
            fname to be used. If None ('*' or '00...' must be in fname), will
            take all images. Can be an integer list of file numbers or can be a
            string with hyphens representing ranges, e.g., '1, 3, 5, 9-15'.

    Returns:
        file_list (list): list of full paths to data.
    """
    if not os.path.isdir(dname):
        raise IOError('Directory does not exist!\n{}'.format(dname))
    elif (numbers is None) and ('*' not in fname):
        raise IOError(
            'No file numbers provided and no wildcard (*) in filename')

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
            n_zeros = len(basename) - len(zero_stripped)
            if n_zeros > 0:
                fname = '{}{{:0{}d}}{}'.format(zero_stripped, n_zeros, extn)
            elif '{:0' not in fname:
                raise IOError('bad filename specifier')

        if numbers.__class__ == str:
            numbers = list_to_indices(numbers)

        file_list = []
        for i in numbers:
            in_file = fname.format(i)
            file_list.append(os.path.join(dname, in_file))
    return file_list


def init_pygix(calibration_dict):
    """
    Instantiate a pygix.Transform() class instance.
    All parameters are set from the yaml file.

    Args:
        calibration_dict (dict): dictionary containing all parameters
            for instantiating pygix.Transform().

    Returns:
        pg (object): class instance of pygix.Transform().
    """
    pg = pygix.transform.Transform()
    pg.load(calibration_dict['ponifile'])

    if 'splinefile' in calibration_dict.keys():
        pg.splinefile = calibration_dict['splinefile']
    if 'flatfile' in calibration_dict.keys():
        pg.flatfiles = calibration_dict['flatfile']
    if 'darkfile' in calibration_dict.keys():
        pg.darkfiles = calibration_dict['darkfile']
    if 'maskfile' in calibration_dict.keys():
        pg.maskfile = calibration_dict['maskfile']

    grazing = calibration_dict['grazing_parameters']
    if 'sample_orientation' in grazing.keys():
        pg.sample_orientation = grazing['sample_orientation']
    else:
        pg.sample_orientation = 1

    pg.incident_angle = grazing['incident_angle']
    if 'tilt_angle' in grazing.keys():
        pg.tilt_angle = grazing['tilt_angle']
    return pg


if __name__ == '__main__':
    if len(sys.argv) is not 2:
        print('usage: process.py recipe.yaml')
        sys.exit(0)
    else:
        rp = Processor(sys.argv[1])
        rp.process()
