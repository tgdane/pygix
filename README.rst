pygix
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

----

Please refer to the `wiki <https://github.com/tgdane/pygix/wiki>`_ for more
detailed discussion!

----

Example usage:

.. code-block:: python

    import pygix
    import pygix.plotting as pp

    pg = pygix.Transform()
    pg.load('detector_calibration.poni')
    pg.indcident_angle = 0.2

    # transform image into reciprocal space:
    i, qxy, qz = pg.transform_reciprocal(data)

    pp.implot(i, qxy, qz, xlim=())

..

![Imgur](http://i.imgur.com/Wvy8Efh.png "Example transformed image")


Pygix uses the fiber transformation originally described by Stribeck [1] (based
on earlier work by Polayni [2]), which has recently been formulated for the case
of grazing-incidence scattering [3]. The reciprocal space transformation for the
fiber and grazing-incidence X-ray scattering are thus equivalent, meaning this
python library is generic for both classes of experiments.

Pygix is heavily based on the pyFAI library for conventional 2D transmission
azimuthal integration [4].


References:
----
1.    Stribeck and Nöchel, J. Appl. Crystallogr., (2009), 42, 295–301
2.    Polanyi, Z. Physik, (1921), 7, 149-180
3.    Lilliu and Dane, 	arXiv:1511.06224 [cond-mat.soft]
4.    Ashiotis, Deschildre, Nawaz, Wright, Karkoulis,
      Picca and Kieffer, J. Appl. Crystallogr., 2015, 48, 510–519
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
* pygix was written by Thomas Dane.
* The derivation of the reciprocal space transformation was done in collaboration with Samuele Lilliu.
* pygix relies heavily on the pyFAI library, and much assistance has been provided by `Jerome Kieffer <https://github.com/kif>`_.