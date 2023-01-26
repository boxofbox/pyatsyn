# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyatsyn Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
# <Oscar Pablo Di Liscia, Pete Moss, and Juan Pampin>


"""
TODO

Attributes
----------
TODO
"""

from math import remainder, tau
from numpy import zeros, matmul

ATS_DEFAULT_SAMPLING_RATE = 44100

def phase_interp_linear(freq_0, freq_t, pha_0, t):
    """Function to compute linear phase interpolation

    Assumes smooth linear interpolation, where the average frequency dictates phase rate estimate 
    from the relative time 0 to time t.

    Parameters
    ----------
    freq_0 : float
        initial frequency (in Hz)
    freq_t : float
        frequency at time t (in Hz)
    pha_0 : float
        initial phase (in radians)
    t : float
        time (in s) from freq_0

    Returns
    -------
    float
        the phase (in radians) at relative time t
    """
    # assuming smooth linear interpolation the average frequency dictates phase rate estimate
    freq_est = (freq_t + freq_0) / 2
    new_phase = pha_0 + (tau * freq_est * t)
    return remainder(new_phase, tau) # NOTE: IEEE remainder


def phase_interp_cubic(freq_0, freq_t, pha_0, pha_t, i_samps_from_0, samps_from_0_to_t, sample_rate):
    """Function to interpolate phase using cubic polynomial interpolation

    Uses cubic interpolation to determine and intermediate phase within the curve linking 
    a particular frequency and phase at relative time 0, to a frequency and phase at time t. 

    The basis for this method is credited to:

        MR. McAulay and T. Quatieri, "Speech analysis/Synthesis based on a 
        sinusoidal representation," in IEEE Transactions on Acoustics, 
        Speech, and Signal Processing, vol. 34, no. 4, pp. 744-754, 
        August 1986
        
        `doi: 10.1109/TASSP.1986.1164910 <https://doi.org/10.1109/TASSP.1986.1164910>`_.

    Parameters
    ----------
    freq_0 : float
        initial frequency (in Hz)
    freq_t : float
        frequency at time t (in Hz)
    pha_0 : float
        initial phase (in radians)
    pha_t : float
        phase at time t (in radians)
    i_samps_from_0 : int
        relative sample index `i` to interpolate at
    samps_from_0_to_t : int
        distance (in samples) from 0 to t
    sample_rate : int
        sampling rate (in samps/s)

    Returns
    -------
    float
        the modeled phase (in radians) at sample `i`
    """ 
    freq_to_radians_per_sample = tau / sample_rate

    alpha_beta_coeffs = zeros([2,2], "float64")
    alpha_beta_coeffs[0][0] = 3 / (samps_from_0_to_t**2)
    alpha_beta_coeffs[0][1] = -1 / samps_from_0_to_t
    alpha_beta_coeffs[1][0] = -2 / (samps_from_0_to_t**3)
    alpha_beta_coeffs[1][1] = 1 / (samps_from_0_to_t**2)
    alpha_beta_terms = zeros([2,1],"float64")

    half_T = samps_from_0_to_t / 2

    w_0 = freq_0 * freq_to_radians_per_sample
    w_t = freq_t * freq_to_radians_per_sample

    M = round((((pha_0 + (w_0 * samps_from_0_to_t) - pha_t) + (half_T * (w_t - w_0))) / tau))
    alpha_beta_terms[0] = pha_t - pha_0 - (w_0 * samps_from_0_to_t) + (tau * M)
    alpha_beta_terms[1] = w_t - w_0
    alpha, beta = matmul(alpha_beta_coeffs, alpha_beta_terms)
    return pha_0 + (w_0 * i_samps_from_0) + (alpha * (i_samps_from_0**2)) + (beta * i_samps_from_0**3)
