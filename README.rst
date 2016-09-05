pygix - Reduction of 2D grazing-incidence and fiber X-ray scattering data
====

Pygix is a generic python library for performing reduction of 
grazing-incidence and fiber X-ray scattering data.

----

The pygix library has been developed to reduce 2D X-ray scattering data for
experiments recorded in grazing-incidence (GISAXS, GIWAXS, GIXRD, GIXD,
collectively referred to as GIXS) and fiber diffraction modes. Both 2D image
projections and 1D line profile extraction are handled. The package is designed
to be as generic as possible, i.e., it makes no assumptions about the data
(passed by the user as numpy arrays) and all detector and geometric corrections
can be handled at the point of data reduction.

Example usage:

.. code-block:: python

    import pygix
    
    pg = pygix.Transform()
    pg.load('detector_calibration.poni')
    pg.indcident_angle = 0.2
    
    i, qxy, qz = pg.transform_reciprocal(data)
..

Pygix uses the fiber transformation originally described by Stribeck [1]_ (based
on earlier work by Polayni [2]_), which has recently been formulated for the case
of grazing-incidence scattering [3]_. The reciprocal space transformation for the
fiber and grazing-incidence X-ray scattering are thus equivalent, meaning this
python library is generic for both classes of experiments.

Pygix is heavily based on the pyFAI library for conventional 2D transmission
azimuthal integration [4]_.


References:
----
.. [1] N. Stribeck and U. Nöchel, J. Appl. Crystallogr., (2009), 42, 295–301.
   [2] M. Polanyi, Z. Physik, (1921), 7, 149-180
   [3] S. Lilliu and T. Dane, 	arXiv:1511.06224 [cond-mat.soft]
   [4] G. Ashiotis, A. Deschildre, Z. Nawaz, J. P. Wright, D. Karkoulis, F. E.
       Picca and J. Kieffer, J. Appl. Crystallogr., 2015, 48, 510–519.
       (https://github.com/silx-kit/pyFAI/)

Installation
----
Download the repository are run setup.py::

    git clone http://github.com/tgdane/pygix.git
    cd pygix
    python setup.py install

..

Credits
----
