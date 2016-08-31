pygix - Reduction of 2D grazing-incidence and fibre X-ray scattering data
====

Pygix is a generic python library for performing reduction of 
grazing-incidence and fibre X-ray scattering data. 

----

Example usage:

    import pygix
    
    pg = pygix.Transform()
    pg.load('detector_calibration.poni')
    pg.indcident_angle = 0.2

    i, qxy, qz = pg.transform_reciprocal(data)