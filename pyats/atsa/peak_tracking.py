# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyats Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
# <Oscar Pablo Di Liscia, Pete Moss and Juan Pampin>


"""TODO Summary

TODO About

"""

from heapq import heappop, heappush
from queue import SimpleQueue
from math import tau, remainder
from numpy import zeros, matmul


def update_track_averages(tracks, track_length, frame_n, analysis_frames, beta = 0.0):
    """Function to TODO

    TODO updates the list of current <tracks>
    we use <track_length> frames of memory to update average amp, frq, and smr of the tracks
    the function returns None, as the tracks are updated directly

    Parameters
    ----------
    tracks : Iterable[:obj:`~pyats.ats_structure.AtsPeak`]
        TODO
    track_length : int
        TODO
    frame_n : int
        TODO
    analysis_frames : 
        TODO
    beta : float, optional
        TODO (default: 0.0)

    """        
    frames = min(frame_n, track_length)
    first_frame = frame_n - frames

    for tk in tracks:
        track = tk.track
        frq_acc = 0
        f = 0
        amp_acc = 0
        a = 0
        smr_acc = 0
        s = 0
        last_frq = 0
        last_amp = 0
        last_smr = 0
        
        for i in range(first_frame, frame_n):

            l_peaks = analysis_frames[i]
            peak = find_track_in_peaks(track, l_peaks)

            if peak is not None:
                if peak.frq > 0.0:
                    last_frq = peak.frq
                    frq_acc += peak.frq
                    f += 1
                if peak.amp > 0.0:
                    last_amp = peak.amp
                    amp_acc += peak.amp
                    a += 1
                if peak.smr > 0.0:
                    last_smr = peak.smr
                    smr_acc += peak.smr
                    s += 1

        if f > 0:
            tk.frq = ((1 - beta) * (frq_acc / f)) + (beta * last_frq)
        if a > 0:
            tk.amp = ((1 - beta) * (amp_acc / a)) + (beta * last_amp)
        if s > 0:
            tk.smr = ((1 - beta) * (smr_acc / s)) + (beta * last_smr)    


def find_track_in_peaks(track, peaks):
    """Function to TODO

    TODO 

    Parameters
    ----------
    tracks : :obj:`~pyats.ats_structure.AtsPeak`
        TODO
    peaks : Iterable[:obj:`~pyats.ats_structure.AtsPeak`]
        TODO

    Returns
    -------
    :obj:`~pyats.ats_structure.AtsPeak`
        TODO
    """ 
    for pk in peaks:
        if track == pk.track:
            return pk
    return None


def peak_tracking(tracks, peaks, frame_n, analysis_frames, sample_rate, hop_size, frequency_deviation = 0.45, SMR_continuity = 0.0, min_gap_length = 1):
    """Function to TODO

    TODO 
    adaptation of the Gale-Shapley algorithm for stable matching of peaks_a and peaks_b
    for matched peaks, track numbers are updated
    tracker is gap-size aware, and will monitor 'slept' tracks within gap distance as candidates
    linear interpolation will be used to fill the gaps
    return value is None, function will update tracks, peaks, and analysis_frames directly

    Parameters
    ----------
    tracks : Iterable[:obj:`~pyats.ats_structure.AtsPeak`]
        TODO
    peaks : Iterable[:obj:`~pyats.ats_structure.AtsPeak`]
        TODO
    frame_n : int
        TODO
    analysis_frames : 
        TODO
    sample_rate : int 
        TODO
    hop_size : int 
        TODO
    frequency_deviation : float, optional
        TODO (default: 0.45)
    SMR_continuity : float, optional
        TODO (default: 0.0)
    min_gap_length : int
        TODO (default: 1)
    """  
    # state for costs
    peak_costs = [[] for _ in peaks]

    # calculate costs for valid peak/track pairs
    for tk_ind, tk in enumerate(tracks):
        if tk.asleep_for is None or tk.asleep_for < min_gap_length:
            for pk_ind, pk in enumerate(peaks):
                if are_valid_candidates(tk, pk, frequency_deviation):
                    cost = peak_dist(tk, pk, SMR_continuity)
                    heappush(peak_costs[pk_ind], MatchCost(cost, tk_ind))
    
    # perform Gale-Shapley stable matching
    peak_queue = SimpleQueue()
    for ind in range(len(peaks)):
        peak_queue.put(ind)
    
    unmatched_peaks = []
    track_matches = [None for _ in tracks]

    while(not peak_queue.empty()):
        pk_ind = peak_queue.get()
        
        made_match = False
        while(len(peak_costs[pk_ind]) > 0):
            tk = heappop(peak_costs[pk_ind])
            # if the track is unmatched
            if track_matches[tk.index] is None:
                # update match
                track_matches[tk.index] = MatchCost(tk.cost, pk_ind)
                made_match = True
                break
            # prior match is more costly, re-match
            elif track_matches[tk.index].cost > tk.cost:
                # kick previous match back onto queue
                peak_queue.put(track_matches[tk.index].index)
                # update match
                track_matches[tk.index] = MatchCost(tk.cost, pk_ind)
                made_match = True
                break
        
        # if we ran out of candidates or no match was made, the peak is unmatched
        if not made_match:
            unmatched_peaks.append(pk_ind)

    unmatched_tracks = []

    # process matches, fixing gaps
    for tk_ind, tk_match in enumerate(track_matches):
        if tk_match is None:
            unmatched_tracks.append(tk_ind)
        else:
            # update matched peak with track number
            pk = peaks[tk_match.index]
            pk.track = tk_ind

            tk = tracks[tk_ind]
            # handle gap if any, with linear interpolation
            if tk.asleep_for is not None:                
                interp_range = tk.asleep_for + 1                                              
                tk.duration += interp_range
                pk.duration = tk.duration
                frq_step = tk.frq - pk.frq
                amp_step = tk.amp - pk.amp
                smr_step = tk.smr - pk.smr                
                for i in range(1, interp_range): # NOTE: we'll walk backward from frame_n
                    new_pk = pk.clone()
                    mult = i / interp_range
                    new_pk.frq = (frq_step * mult) + pk.frq
                    new_pk.amp = (amp_step * mult) + pk.amp
                    new_pk.smr = (smr_step * mult) + pk.smr
                    samps_from_0_to_t = (interp_range + 1) * hop_size
                    i_samps_from_0 = hop_size * (interp_range - i)                   
                    new_pk.pha = phase_interp_cubic(tk.frq, pk.frq, tk.pha, pk.pha, i_samps_from_0, samps_from_0_to_t, sample_rate)
                    new_pk.duration -= i
                    analysis_frames[frame_n - i].append(new_pk)
                tk.asleep_for = None                
            else:
                tk.duration += 1
                pk.duration = tk.duration

    # instantiate new tracks for unmatched peaks
    for pk_ind in unmatched_peaks:
        peaks[pk_ind].track = len(tracks)
        tracks.append(peaks[pk_ind].clone())

    # sleep or update sleep for unmatched tracks
    for tk_ind in unmatched_tracks:
        if tracks[tk_ind].asleep_for is None:
            tracks[tk_ind].asleep_for = 1
        else:
            tracks[tk_ind].asleep_for += 1


def are_valid_candidates(candidate1, candidate2, deviation):
    """Function to TODO

    TODO

    Parameters
    ----------
    candidate1 : :obj:`~pyats.ats_structure.AtsPeak`
        TODO
    candidate2 : :obj:`~pyats.ats_structure.AtsPeak`
        TODO
    deviation : float
        TODO

    Returns
    -------
    float
        TODO
    """ 
    min_frq = min(candidate1.frq, candidate2.frq)
    return abs(candidate1.frq - candidate2.frq) <= 0.5 * min_frq * deviation


def peak_dist(pk1, pk2, alpha):
    """Function to TODO

    TODO 

    Parameters
    ----------
    pk1 : :obj:`~pyats.ats_structure.AtsPeak`
        TODO
    pk1 : :obj:`~pyats.ats_structure.AtsPeak`
        TODO
    alpha : float
        TODO

    Returns
    -------
    float
        TODO
    """ 
    return (abs(pk1.frq - pk2.frq) + (alpha * abs(pk1.smr - pk2.smr))) / (alpha + 1.0)


def phase_interp(freq_0, freq_t, pha_0, t):
    """Function to TODO

    TODO returns the phase (-pi,pi] at time t
    given that the freq linearly interpolates from
    freq_0, with phase pha_0 at time 0 to freq_t at time t

    Parameters
    ----------
    freq_0 : float
        TODO
    freq_t : float
        TODO
    pha_0 : float
        TODO
    t : float
        TODO

    Returns
    -------
    float
        TODO
    """
    # assuming smooth linear interpolation the average frequency dictates phase rate estimate
    freq_est = (freq_t + freq_0) / 2
    new_phase = pha_0 + (tau * freq_est * t)
    return remainder(new_phase, tau) # NOTE: IEEE remainder


def phase_interp_cubic(freq_0, freq_t, pha_0, pha_t, i_samps_from_0, samps_from_0_to_t, sample_rate):
    """Function to TODO

    TODO 
    for cubic polynomial interpolation of phase
    credit: McAulay & Quatieri (1986)

    Parameters
    ----------
    freq_0 : float
        TODO
    freq_t : float
        TODO
    pha_0 : float
        TODO
    pha_t : float
        TODO
    i_samps_from_0 : int
        TODO
    samps_from_0_to_t : int
        TODO
    sample_rate : int
        TODO

    Returns
    -------
    float
        TODO
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

 
class MatchCost():
    """TODO

    """ 
    def __init__(self, cost, index):
        self.cost = cost
        self.index = index
    
    def __repr__(self):
        return f"to index: {self.index} at cost: {self.cost}"
    
    def __lt__(self, other):
        return self.cost < other.cost