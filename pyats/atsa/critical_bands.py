# -*- coding: utf-8 -*-
"""Critical Bands and Signal-to-Mask Ratio Evaluation

This module is used to evaluate critical band masking for signal-to-mask ratio calculations


pyats Copyright (c) <2023>, <Johnathan G Lyon>
All rights reserved.

Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>, <Oscar Pablo
Di Liscia, Pete Moss and Juan Pampin>

This source code is licensed under the BSD-style license found in the
LICENSE.rst file in the root directory of this source tree. 

Attributes
----------
ATS_CRITICAL_BAND_EDGES : ndarray
    1D array containing 26 `float64` frequencies that distinguish the default 25 critical bands
"""

from numpy import log10, array

from pyats.atsa.utils import amp_to_db_spl


ATS_CRITICAL_BAND_EDGES = array([0.0,100.0,200.0,300.0, 400.0,
                                510.0, 630.0, 770.0, 920.0, 1080.0,
                                1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 
                                2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 
                                6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 
                                20000.0], dtype="float64")


def evaluate_smr(peaks, slope_l = -27.0, delta_db = -50):    
    """Function to evaluate signal-to-mask ratio for the given peaks

    This function evaluates masking values (SMR) for peaks in list `peaks`
    The parameters will be use to calculate the triangle mask 

    # TODO
    [slope_l] are the slope of left side of the mask
    in dBs/bark, <delta_db> is the dB treshold for
    the masking curves (must be <= 0dB) 

    Parameters
    ----------
    peaks : Iterable[AtsPeaks]
        An iterable collection of AtsPeaks that will have their `smr` attributes updated
    slope_l : float, optional
        A float to dictate the slope of the left side of the mask (default: -27.0)
    delta_db : float, optional
        A float (in dB) that sets the amplitude threshold for the masking curves
        Must be (<= 0dB) (default: -50)

    Raises
    ------
    ValueError
        If `delta_db` is not less than or equal to 0.

    """
    if delta_db > 0:
        raise ValueError("delta_db must be <= 0")

    n_peaks = len(peaks)
    if n_peaks == 1:
        peaks[0].smr = amp_to_db_spl(peaks[0].amp)    
    else:
        for p in peaks:
            p.barkfrq = frq_to_bark(p.frq)
            p.db_spl = amp_to_db_spl(p.amp)
            p.slope_r = compute_slope_r(p.db_spl, slope_l)        

        for maskee_ind, maskee in enumerate(peaks):

            for masker_ind in [ i for i in range(n_peaks) if i != maskee_ind]:
                masker = peaks[masker_ind]                
                
                mask_term = masker.db_spl + delta_db + (masker.slope_r * abs(maskee.barkfrq - masker.barkfrq))
                if mask_term > maskee.smr:
                    maskee.smr = mask_term

            maskee.smr = maskee.db_spl - maskee.smr


def frq_to_bark(freq):
    """Function to convert frequency from Hz to bark scale

    # TODO

    Parameters
    ----------


    """
    if freq <= 0.0:
        return 0.0
    elif freq <= 400.0:
        return freq * 0.01
    elif freq >= 20000.0:
        return None
    else:
        band = find_band(freq)
        low = ATS_CRITICAL_BAND_EDGES[band]
        high = ATS_CRITICAL_BAND_EDGES[band+1]
        return 1 + band + abs(log10(freq/low) / log10(low/high))


def find_band(freq):
    """Function to ...

    # TODO

    Parameters
    ----------


    """
    for ind in range(len(ATS_CRITICAL_BAND_EDGES)-2,0,-1):
        if freq > ATS_CRITICAL_BAND_EDGES[ind]:
            return ind
    return 0


def compute_slope_r(masker_amp_db, slope_l = -27.0):
    """Function to ...

    # TODO

    Parameters
    ----------


    """
    """
    computes the masker slope toward high frequencies
    depends on the levle of the masker
    """
    return slope_l + (max(masker_amp_db - 40.0, 0.0) * 0.37)