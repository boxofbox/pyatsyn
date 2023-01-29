# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyatsyn Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
# <Oscar Pablo Di Liscia, Pete Moss, and Juan Pampin>


"""
Peak Tracking algorithms to assemble spectral trajectories

Peaks issued by the peak detection algorithm need to be connected and 
translated into spectral trajectories. This process involves the evaluation 
of the possible candidates to continue trajectories on a frame-by-frame basis. 

This is done using tracks that keep information of recent average values for each 
of the trajectory parameters. The length of the tracks is adjustable and has to 
be tuned depending on the characteristics of the analyzed sound. 

A Gale-Shapley stable matching algorithm is used to determine the best candidate pair using a the cost criteria:

    :math:`cost = \\frac{|P_{freq} - T_{freq}| + \\alpha * |P_{smr} - T_{smr}|}{1 + \\alpha}`
    
where :math:`P_{freq}` is the candidate peak frequency, and :math:`P_{smr}` its SMR, :math:`T_{freq}` 
is the track frequency, and :math:`T_{smr}` its SMR, both averaged over the track length (typically 3 frames). 
:math:`\\alpha` is a coefficient controlling how much the SMR deviation affects the cost. 

The use of the SMR continuation as a parameter for the peak tracking process is based upon 
psychoacoustic temporal masking phenomena. Conceptually, we assume that masking profiles of 
stable sinusoidal trajectories can only evolve at slow rate (no sudden changes). This is true for 
analysis performed with hop sizes between 10 and 50 milliseconds, which is comparable to the 
average duration of pre- and post-making effects.

New tracks get created from orphan peaks (the ones that were not incorporated to any existing tracks), 
and tracks which couldn't find continuing peaks are set to sleep.

"""

from heapq import heappop, heappush
from queue import SimpleQueue
from pyatsyn.ats_utils import phase_interp_cubic
from ats_structure import MatchCost


def update_track_averages(tracks, track_length, frame_n, analysis_frames, beta = 0.0):
    """Function to update running averages of recent peaks

    Using the list of current tracks, we use `track_length` frames to look back and update, 
    the average amp, frq, and smr values for the tracks. 
    
    NOTE: Tracks are updated directly without return value.

    Parameters
    ----------
    tracks : Iterable[:obj:`~pyatsyn.ats_structure.AtsPeak`]
        iterable of tracks to update
    track_length : int
        how far back in time (in frames) to start average calculations
    frame_n : int
        the current frame
    analysis_frames : Iterable[Iterable[:obj:`~pyatsyn.ats_structure.AtsPeak`]]
        a running collection storing the :obj:`~pyatsyn.ats_structure.AtsPeak` objects at each frame in time
    beta : float, optional
        TOadditional bias for the immediately prior frames values when calculating smoothing trajectories (default: 0.0)
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
    """Function to search a the first peak found tagged a given track ind

    Parameters
    ----------
    track : int
        the track index to search for
    peaks : Iterable[:obj:`~pyatsyn.ats_structure.AtsPeak`]
        a collection of :obj:`~pyatsyn.ats_structure.AtsPeak` to search in

    Returns
    -------
    :obj:`~pyatsyn.ats_structure.AtsPeak`
        the first :obj:`~pyatsyn.ats_structure.AtsPeak` found in `peaks` that has a .track attribute matching `track`.
        If no matches are found, None is returned.
    """ 
    for pk in peaks:
        if track == pk.track:
            return pk
    return None


def peak_tracking(tracks, peaks, frame_n, analysis_frames, sampling_rate, hop_size, frequency_deviation = 0.1, SMR_continuity = 0.0, min_gap_length = 1):
    """Core function to coordinate peak tracking

    This function coordinates the matching of new peaks with existing tracks using 
    an adaptation of the Gale-Shapley algorithm for stable matching. The algorithm is 
    gap-size aware and will monitor 'slept' tracks within the minimum gap distance as 
    candidates. Linear interpolation is used to fill the gaps for frequency and amplitude,
    and a cubic polynomial interpolation for phase.
    
    NOTE: Tracks, peaks, and analysis_frames are updated directly.

    Parameters
    ----------
    tracks : Iterable[:obj:`~pyatsyn.ats_structure.AtsPeak`]
        collection of established tracks
    peaks : Iterable[:obj:`~pyatsyn.ats_structure.AtsPeak`]
        collection of candidate peaks to match
    frame_n : int
        the current frame
    analysis_frames : Iterable[Iterable[:obj:`~pyatsyn.ats_structure.AtsPeak`]]
        a running collection storing the :obj:`~pyatsyn.ats_structure.AtsPeak` objects at each frame in time
    sampling_rate : float
        the sampling rate (in samples / s)
    hop_size : int 
        the inter-frame distance (in samples)
    frequency_deviation : float, optional
        maximum relative frequency deviation used to constrain peak tracking matches (default: 0.1)
    SMR_continuity : float, optional
        percentage of SMR to use in cost calculations during peak tracking (default: 0.0)
    min_gap_length : int
        tracked partial gaps longer than this (in frames) will not be interpolated (default: 1)
    """  
    # state for costs
    peak_costs = [[] for _ in peaks]

    # calculate costs for valid peak/track pairs
    for tk_ind, tk in enumerate(tracks):
        if tk.asleep_for is None or tk.asleep_for < min_gap_length:
            for pk_ind, pk in enumerate(peaks):
                if are_valid_peak_candidates(tk, pk, frequency_deviation):
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
                    new_pk.pha = phase_interp_cubic(tk.frq, pk.frq, tk.pha, pk.pha, i_samps_from_0, samps_from_0_to_t, sampling_rate)
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


def are_valid_peak_candidates(candidate1, candidate2, deviation):
    """Function to determine if the distance between two peaks are within the relative deviation constraint

    Peaks are valid candidates for pairing if their absolute distance is smaller than the frequency deviation 
    multiplied by the lower of the candidate's frequencies.

    Parameters
    ----------
    candidate1 : :obj:`~pyatsyn.ats_structure.AtsPeak`
        a candidate peak
    candidate2 : :obj:`~pyatsyn.ats_structure.AtsPeak`
        a candidate peak
    deviation : float
        relative frequency deviation

    Returns
    -------
    bool
        True if the candidates are within constrained range, False otherwise.
    """ 
    min_frq = min(candidate1.frq, candidate2.frq)
    return abs(candidate1.frq - candidate2.frq) <= 0.5 * min_frq * deviation


def peak_dist(pk1, pk2, alpha):
    """Function to calculate peak frequency distance

    This function is used to calculate the cost for the peak matching algorithm 
    and allows for psychoacoustic biasing of the calculation:

        :math:`dist = \\frac{|P1_{freq} - P2_{freq}| + \\alpha * |P1_{smr} - P2_{smr}|}{1 + \\alpha}`

    where :math:`P\#_{freq}` is the peak's frequency, and :math:`P\#_{smr}` its SMR.  
    :math:`\\alpha` is a coefficient controlling how much the SMR deviation affects the distance.   

    Parameters
    ----------
    pk1 : :obj:`~pyatsyn.ats_structure.AtsPeak`
        a candidate peak
    pk1 : :obj:`~pyatsyn.ats_structure.AtsPeak`
        a candidate peak
    alpha : float
        percent of SMR to use to bias the result

    Returns
    -------
    float
        the frequency distance (in Hz) between the peaks
    """ 
    return (abs(pk1.frq - pk2.frq) + (alpha * abs(pk1.smr - pk2.smr))) / (alpha + 1.0)
