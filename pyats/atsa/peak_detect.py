# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyats Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>, <Oscar Pablo
# Di Liscia, Pete Moss and Juan Pampin>


"""Single-Frame Peak Detection from FFT Data

Functions to process FFT data and extract peaks

"""

from numpy import pi
from math import tau

from pyats.ats_structure import AtsPeak
from pyats.atsa.utils import amp_to_db, db_to_amp


def peak_detection (fftfreqs, fftmags, fftphases, 
                    lowest_bin=None, highest_bin=None, lowest_magnitude=None):
    """Function to detect peaks from FFT data

    TODO

    Parameters
    ----------
    fftfreqs : ndarray[float64]
        TODO
    fftmags : ndarray[float64]
        TODO
    fftphases : ndarray[float64]
        TODO
    lowest_bin : int, optional
        TODO (default: None)
    highest_bin : int, optional
        TODO (default: None)
    lowest_magnitude : float, optional
        TODO (default: None)

    Returns
    -------
    list[:obj:`~pyats.ats_structure.AtsPeak`]
        TODO
    """
    peaks = []

    N = highest_bin
    if N is None:
        N = fftfreqs.size - 1
    
    first_bin = lowest_bin
    if first_bin is None or first_bin < 1:
        first_bin = 1
    
    frqs = fftfreqs[first_bin-1:N + 1]
    mags = fftmags[first_bin-1:N + 1]
    phs = fftphases[first_bin-1:N + 1]
    fq_scale = frqs[1] - frqs[0]

    for k in range(1, N - first_bin):

        left = mags[k-1]
        center = mags[k]
        right = mags[k+1]

        if center > lowest_magnitude and center > right and center > left:
            pk = AtsPeak()
            offset, pk.amp = parabolic_interp(left, center, right)
            pk.frq = frqs[k] + (fq_scale * offset)
            if (offset > 0):
                pk.pha = phase_correct(phs[k-1], phs[k], offset)
            else:
                pk.pha = phase_correct(phs[k], phs[k+1], offset)
            peaks.append(pk)

    return peaks


def parabolic_interp(alpha, beta, gamma):
    """Function to TODO

    TODO
    does parabolic interpolation of 3 points
    returns the x offset and height
    of the interpolated peak  

    Parameters
    ----------
    alpha : float
        TODO
    beta : float
        TODO
    gamma : float
        TODO

    Returns
    -------
    offset : float
        TODO
    height : float
        TODO
    """
    dB_alpha = amp_to_db(alpha)
    dB_beta = amp_to_db(beta)
    dB_gamma = amp_to_db(gamma)
    dB_alpha_minus_gamma = dB_alpha - dB_gamma
    offset = 0.5 * dB_alpha_minus_gamma / (dB_alpha + dB_gamma + (-2 * dB_beta))
    height = db_to_amp(dB_beta - (0.25 * dB_alpha_minus_gamma * offset))
    return offset, height

def phase_correct(left, right, offset):
    """Function to TODO

    TODO
    angular interpolation 

    Parameters
    ----------
    left : float
        TODO
    right : float
        TODO
    offset : float
        TODO

    Returns
    -------
    float
        TODO
    """
    if left - right > 1.5 * pi:
        return (left + (offset * (right - left + tau)))
    elif right - left > 1.5 * pi:
        return (left + (offset * (right - left - tau)))
    else:
        return (left + (offset * (right - left)))