import numpy as np
from numpy import pi, sqrt, deg2rad, rad2deg, cos, sin, arcsin, arccos, arctan2


def calc_unit_cell_vol(a, b, c, alpha, beta, gamma):
    """
    Calculate the volume of a unit cell. Lattice parameters
    a, b and c should be passed in units of Angstrom. Lattice
    parameters alpha, beta and gamma should be passed in units
    of degrees.

    Args:
        a (float) : lattice parameter a
        b (float) : lattice parameter b
        c (float) : lattice parameter c
        alpha (float) : lattice parameter alpha
        beta (float) : lattice parameter beta
        gamma (float) : lattice parameter gamma

    Returns:
        vol (float) : volume (cubed Angstrom) of unit cell.
    """
    volume = a * b * c * (sqrt(1 - (cos(alpha) ** 2) - (cos(beta) ** 2) -
                               (cos(gamma) ** 2) + (2 * cos(alpha) * cos(beta) *
                                                    cos(gamma))))
    return volume


def calc_reciprocal_lattice(a, b, c, alpha, beta, gamma):
    """
    Calculates the reciprocal lattice parameters: a*, b*, c*, alpha*, beta*
    and gamma*.
    """
    volume = calc_unit_cell_vol(a, b, c, alpha, beta, gamma)
    reciprocal_lattice = np.zeros(6)
    reciprocal_lattice[0] = 2 * pi * (b * c * sin(alpha)) / volume
    reciprocal_lattice[1] = 2 * pi * (a * c * sin(beta)) / volume
    reciprocal_lattice[2] = 2 * pi * (a * b * sin(gamma)) / volume
    reciprocal_lattice[3] = arcsin(
        volume / (a * b * c * sin(beta) * sin(gamma)))
    reciprocal_lattice[4] = arcsin(
        volume / (a * b * c * sin(alpha) * sin(gamma)))
    reciprocal_lattice[5] = arcsin(
        volume / (a * b * c * sin(alpha) * sin(beta)))
    return reciprocal_lattice


def calc_reciprocal_lattice_vectors(reciprocal_lattice, alpha, c):
    """ """
    rl = reciprocal_lattice
    rl_vectors = np.array((
        (rl[0], 0, 0),
        (rl[1] * cos(rl[5]), rl[1] * sin(rl[5]), 0),
        (rl[2] * cos(rl[4]), -rl[2] * sin(rl[4]) * cos(alpha), 2 * pi / c)
        ))
    return rl_vectors


def calc_orientation_vector(rl_vectors, orientation):
    """ """
    rl_a = rl_vectors[0]
    rl_b = rl_vectors[1]
    rl_c = rl_vectors[2]
    g_hkl = orientation[0] * rl_a + orientation[1] * rl_b + orientation[2] * \
                                                            rl_c
    return g_hkl


def calc_rotation_angles(g_hkl):
    """ """
    phi = arctan2(g_hkl[1], g_hkl[0])
    chi = arccos(g_hkl[2] / sqrt(g_hkl[0] ** 2 + g_hkl[1] ** 2 + g_hkl[2] ** 2))
    return phi, chi


def calc_rotation_matrix(phi, chi):
    """ """
    rot1 = np.array(((cos(chi), 0, sin(-chi)),
                     (0, 1, 0),
                     (sin(chi), 0, cos(chi))))
    rot2 = np.array(((cos(phi), sin(phi), 0),
                     (-sin(phi), cos(phi), 0),
                     (0, 0, 1)))
    rotation_matrix = np.dot(rot1, rot2)
    return rotation_matrix


def orient_reciprocal_lattice_vectors(rl_vectors, rotation_matrix):
    """ """
    rl_A = rl_vectors[0]
    rl_B = rl_vectors[1]
    rl_C = rl_vectors[2]
    RA = np.dot(rotation_matrix, rl_A)
    RB = np.dot(rotation_matrix, rl_B)
    RC = np.dot(rotation_matrix, rl_C)
    return RA, RB, RC


def calc_ub_matrix(lattice_paramters, orientation):
    """ """
    a, b, c, alpha, beta, gamma = lattice_paramters

    reciprocal_lattice = calc_reciprocal_lattice(a, b, c, alpha, beta, gamma)
    rl_vectors = calc_reciprocal_lattice_vectors(reciprocal_lattice, alpha, c)
    g_hkl = calc_orientation_vector(rl_vectors, orientation)
    phi, chi = calc_rotation_angles(g_hkl)
    rotation_matrix = calc_rotation_matrix(phi, chi)
    ub_matrix = orient_reciprocal_lattice_vectors(rl_vectors, rotation_matrix)
    return ub_matrix


def calc_reflections(lattice_parameters, orientation, hkl):
    """
    Calculate the position of reflections in reciprocal space based
    on lattice parameters and orientation for reflections with
    Miller indices of hkl.

    Args:
        lattice_parameters (list) : (a, b, c, alpha, beta, gamma). Length
            in units of Angstrom or nm, angles in degrees.
        orientation (list) : Reflection that is aligned normal to the surface
            plane defining the crystallographic orientation relative to the
            substrate.
        hkl : (np.array) : (n x 3) array of hkl reflections to be calculated.

    Returns:
        qr (np.array) : in-plane coordinates of reflections.
        qz (np.array) : out-of-plane coordinates of reflections.
    """
    # if len(lattice_parameters) is not 6:
    #     raise RuntimeError('lattice parameters should have length 6')
    # if len(orientation) is not 3:
    #     raise RuntimeError('orientation must have length 3 (hkl)')
    # if hkl.shape[1] is not 3:
    #     raise RuntimeError('hkl must have 3 columns')

    ub_a, ub_b, ub_c = calc_ub_matrix(lattice_parameters, orientation)

    qr_calc = np.zeros(hkl.shape[0])
    qz_calc = np.zeros(hkl.shape[0])

    for i in range(hkl.shape[0]):
        q = hkl[i, 0] * ub_a + hkl[i, 1] * ub_b + hkl[i, 2] * ub_c
        qr_calc[i] = sqrt(q[0] ** 2 + q[1] ** 2)
        qz_calc[i] = q[2]
    return qr_calc, qz_calc
