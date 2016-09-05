pygix - Reduction of 2D grazing-incidence and fibre X-ray scattering data
====

Pygix is a generic python library for performing reduction of 
grazing-incidence and fibre X-ray scattering data. 

----

Example usage:

.. code-block:: python

    import pygix
    
    pg = pygix.Transform()
    pg.load('detector_calibration.poni')
    pg.indcident_angle = 0.2
    
    i, qxy, qz = pg.transform_reciprocal(data)
..

Installation
----
Download the repository::

    git clone http://github.com/tgdane/pygix.git
    cd pygix
    python setup.py install

