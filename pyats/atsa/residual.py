# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyats Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>, <Oscar Pablo
# Di Liscia, Pete Moss and Juan Pampin>


"""TODO Summary

TODO About

"""

from numpy import zeros, mean, roll, arange, absolute, asarray
from numpy.fft import fft
import soundfile as sf

from pyats.ats_synth import synth

from pyats.atsa.utils import next_power_of_2, db_to_amp, ATS_NOISE_THRESHOLD
from pyats.atsa.critical_bands import ATS_CRITICAL_BAND_EDGES


def compute_residual(   residual_file, 
                                ats_snd, 
                                in_sound,
                                start_sample,
                                end_sample,
                                export_residual = True,
                                ):
    """Function to computes the difference between the ats_snd synthesis and the original sound

    TODO 

    Parameters
    ----------
    residual_file : str
        TODO
    ats_snd : :obj:`~pyats.ats_structure.AtsSound`
        TODO
    in_sound : ndarray[float]
        TODO
    start_sample : int
        TODO
    end_sample : int
        TODO
    export_residual: bool, optional
        TODO (default: True)

    Returns
    -------
    residual : ndarray[float]
        TODO
    """ 
    synthesized = synth(ats_snd)
    residual = in_sound[start_sample:end_sample] - synthesized

    # export residual to audio file
    if export_residual:
        sf.write(residual_file, residual, ats_snd.sampling_rate)

    return residual


def residual_analysis(  residual, 
                        ats_snd, 
                        min_fft_size = 4096,
                        equalize = False,
                        pad_factor = 2,
                        band_edges = None,
                        par_energy = False,
                        verbose = True,                                               
                        ):
    """Function to TODO

    TODO 

    Parameters
    ----------
    residual : ndarray[float]
        TODO
    ats_snd : :obj:`~pyats.ats_structure.AtsSound`
        TODO
    min_fft_size : int, optional
        TODO (default: 4096)
    equalize : bool, optional
        TODO (default: False)
    pad_factor : int, optional
        TODO (default: 2)
    band_edges : TODO
        TODO (default: None)
    par_energy : bool
        TODO (default: False)
    verbose : bool, optional
        TODO (default: True)
    """ 
    hop = ats_snd.frame_size
    M = ats_snd.window_size
    M_over_2 = (M - 1) // 2

    threshold = db_to_amp(ATS_NOISE_THRESHOLD)

    N = residual_N(M, min_fft_size, pad_factor)

    frames = ats_snd.frames
    fft_mag = ats_snd.sampling_rate / N

    # sub band limits
    if band_edges is None:
        band_edges = ATS_CRITICAL_BAND_EDGES
    band_limits = residual_get_band_limits(fft_mag, band_edges)
    n_bands = len(band_limits) - 1
    band_energy = zeros([n_bands,frames],"float64")

    if verbose:
        print(f"Beginning Residual FFT... [frames = {frames}; M = {M}; N = {N}]")
        report_every_x_percent = 10
    report_flag = 0

    fft_mags = None
    t_domain_energy = 0.0
    fil_ptr = -M_over_2

    for frame_n in range(frames):

        if verbose:
            done = frame_n * 100.0 / frames
            if done > report_flag:
                print(f"\t{done:.2f}% complete (residual analysis)")
                report_flag += report_every_x_percent
        
        # padding for window ranges that are out of the input file
        front_pad = 0
        back_pad = 0
        if fil_ptr < 0:
            front_pad = -fil_ptr
        if fil_ptr + M >= residual.size:
            back_pad = fil_ptr + M - residual.size
        
        data = zeros(N, "float64")

        # pull in data
        data[front_pad:M-back_pad] = residual[fil_ptr+front_pad:fil_ptr+M-back_pad]                    
        if equalize:
            # store the time domain energy for equalization
            t_domain_energy = sum(data[front_pad:M-back_pad]**2)
        
        # shift window by half of M so that phases in `data` are relatively accurate to midpoints
        data = roll(data, -M_over_2)
        
        # update file pointer
        fil_ptr += hop

        # DC Block
        data = data - mean(data)

        # FFT
        fd = fft(data)

        fft_mags = absolute(fd) * 2.0

        residual_compute_band_energy(fft_mags, band_limits, band_energy, frame_n)

        if equalize:
            # re-scale frequency band energy to the energy in the time domain                 
            f_domain_energy = sum(band_energy[:,frame_n])
            
            eq_ratio = 1.0
            if f_domain_energy > 0.0:
                eq_ratio = t_domain_energy / f_domain_energy
            band_energy[:,frame_n] *= eq_ratio
            
    # apply noise threshold
    band_energy[band_energy < threshold] = 0.0
    
    # store in ats object
    ats_snd.band_energy = band_energy
    ats_snd.bands = arange(n_bands, dtype='int64')

    if par_energy:
        if verbose:
            print("Transferring noise band energy to partials...")
        band_to_energy(ats_snd, band_edges)
        remove_bands(ats_snd, ATS_NOISE_THRESHOLD)


def residual_N(M, min_fft_size, factor = 2):
    """Function to TODO

    TODO 

    Parameters
    ----------
    M : int
        TODO
    min_fft_size : int
        TODO
    factor : int, optional
        TODOe)

    Returns
    -------
    int
        TODO
    """ 
    if M * factor > min_fft_size:
        return next_power_of_2(M * factor)
    else:
        return next_power_of_2(min_fft_size)


def residual_get_band_limits(fft_mag, band_edges):
    """Function to TODO

    TODO 

    Parameters
    ----------
    fft_mag : float
        TODO
    band_edges : TODO
        TODO

    Returns
    -------
    band_limits : TODO
        TODO
    """ 
    band_limits = zeros(len(band_edges),"int64")
    for ind, band in enumerate(band_edges):
        band_limits[ind] = band / fft_mag
    return band_limits 


def residual_compute_band_energy(fft_mags, band_limits, band_energy, frame_n):
    """Function to TODO

    TODO 

    Parameters
    ----------
    fft_mags : ndarray[float]
        TODO
    band_limits : TODO
        TODO
    band_energy : TODO
        TODO
    frame_n : int
        TODO
    """ 
    for band in range(len(band_limits) - 1):
        low = band_limits[band]
        if low < 0:
            low = 0
        high = band_limits[band + 1]
        if high > fft_mags.size // 2:
            high = fft_mags.size // 2

        band_energy[band][frame_n] = sum(fft_mags[low:high]**2) / fft_mags.size


def band_to_energy(ats_snd, band_edges, use_smr = False):
    """Function to TODO

    TODO 

    Parameters
    ----------
    ats_snd : :obj:`~pyats.ats_structure.AtsSound`
        TODO
    band_edges : TODO
        TODO
    use_smr : bool, optional
        TODO (default: False)
    """     
    bands = len(ats_snd.bands)
    partials = ats_snd.partials
    frames = ats_snd.frames
    par_energy = zeros([partials,frames],"float64")
    partial_ind = 0

    for frame_n in range(frames):
        for band in range(bands):

            # get frame's partials that are within the subband
            lo_frq = band_edges[band]
            hi_frq = band_edges[band + 1]
            par = []
            while (partial_ind < partials):
                check = ats_snd.frq[partial_ind][frame_n]
                if check < lo_frq:
                    partial_ind += 1
                elif check >= lo_frq and check <= hi_frq:
                    par.append(partial_ind)
                    partial_ind += 1
                else:
                    break
            
            if par and ats_snd.band_energy[band][frame_n] > 0.0:
                
                amp_source = ats_snd.amp
                if use_smr:
                    amp_source = ats_snd.smr

                # get the current energy of the subband
                par_amp_sum = 0.0
                for p_ind in par:
                    par_amp_sum += amp_source[p_ind][frame_n]
                
                if par_amp_sum > 0.0:
                    # if the sub-band is active store band energy proportionally among activate partials
                    for p in par:
                        par_energy[p][frame_n] = amp_source[p][frame_n] * ats_snd.band_energy[band][frame_n] / par_amp_sum
                else:
                    # otherwise spread the energy across the inactive partials
                    energy_per_partial = ats_snd.band_energy[band][frame_n] / len(par)
                    for p in par:
                        par_energy[p][frame_n] = energy_per_partial
        
                # clear energy redistributed to partials from band
                ats_snd.band_energy[band][frame_n] = 0.0

    ats_snd.energy = par_energy
            
            
def remove_bands(ats_snd, threshold):
    """Function TODO

    TODO  remove bands from ats_snd that are below threshold (in dB)

    Parameters
    ----------
    ats_snd : :obj:`~pyats.ats_structure.AtsSound`
        TODO
    threshold : float
        TODO
    """ 
    frames = ats_snd.frames
    threshold = db_to_amp(threshold)
    
    valid_bands = []
    for band in ats_snd.bands:
        if mean(ats_snd.band_energy[band][ats_snd.band_energy[band] > 0.0]) >= threshold:
            valid_bands.append(band)
    
    if valid_bands:
        ats_snd.band_energy = ats_snd.band_energy[valid_bands,:]
        ats_snd.bands = asarray(valid_bands, 'int64')
    else:
        ats_snd.band_energy = []
        ats_snd.bands = []