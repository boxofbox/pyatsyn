# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyats Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>, <Oscar Pablo
# Di Liscia, Pete Moss and Juan Pampin>


"""TODO Summary

TODO About

Attributes
----------
MAX_DB_SPL : float
    TODO
ATS_MIN_SEGMENT_LENGTH : int
    TODO
ATS_AMP_THRESHOLD : float
    TODO
ATS_NOISE_THRESHOLD : float
    TODO
"""

from numpy import inf, ceil, log2, log10

###################
# UTILITY CONSTANTS
###################

MAX_DB_SPL = 100.0
ATS_MIN_SEGMENT_LENGTH = 3
ATS_AMP_THRESHOLD = -60
ATS_NOISE_THRESHOLD = -120


###################
# UTILITY FUNCTIONS
###################

def db_to_amp(db):
    """Function to TODO

    TODO
    convert decibels to amplitude

    Parameters
    ----------
    db : float
        TODO

    Returns
    -------
    float
        TODO
    """
    if (db == -inf):
        return 0.0
    return pow(10, (db / 20.0))


def amp_to_db(amp):
    """Function to TODO

    TODO
    convert amplitude to decibels

    Parameters
    ----------
    amp : float
        TODO

    Returns
    -------
    float
        TODO
    """
    return 20 * log10(amp)


def amp_to_db_spl(amp):
    """Function to TODO

    TODO

    Parameters
    ----------
    amp : float
        TODO

    Returns
    -------
    float
        TODO
    """
    return MAX_DB_SPL + amp_to_db(amp)


def next_power_of_2(num):
    """Function to TODO

    TODO return the closest power of 2 integer more than or equal to <num>

    Parameters
    ----------
    num : int
        TODO

    Returns
    -------
    int
        TODO
    """
    return int(2**ceil(log2(num)))


def compute_frames(total_samps, hop):
    """Function to TODO

    TODO
    computes the number of frames in the specified analysis
    we want to have an extra frame at the end to prevent chopping the ending

    Parameters
    ----------
    total_samps : int
        TODO
    hop : int
        TODO

    Returns
    -------
    int
        TODO
    """
    return int(ceil(total_samps / hop)) + 1
        

def optimize_tracks(tracks, analysis_frames, min_segment_length, 
                        amp_threshold, highest_frequency, lowest_frequency):
    """Function to TODO

    TODO

    Parameters
    ----------
    tracks : Iterable[:obj:`~pyats.ats_structure.AtsSound`]
        TODO
    analysis_frames : TODO
        TODO
    min_segment_length : int
        TODO
    amp_threshold : float
        TODO
    highest_frequency : float
        TODO
    lowest_frequency : float
        TODO

    Returns
    -------
    TODO
    """
    if min_segment_length < 1:
        min_segment_length = ATS_MIN_SEGMENT_LENGTH

    # NOTE: amp_threshold is expected in amps
    if amp_threshold == None:
        amp_threshold = db_to_amp(ATS_AMP_THRESHOLD)
    
    tracks_for_removal = set()

    # trim short partials
    for tk in tracks:
        if tk.duration < min_segment_length:
            tracks_for_removal.add(tk.track)
        else:
            # zero amp & frq for averages
            tk.frq = 0.0
            tk.amp = 0.0
    
    # get max & average values (store data on tracks)
    for frame_n in range(len(analysis_frames)):
        for pk in analysis_frames[frame_n]:
            tk_ind = pk.track
            if tk_ind not in tracks_for_removal:
                tk = tracks[tk_ind]
                tk.amp_max = max(tk.amp_max, pk.amp)
                tk.frq_max = max(tk.frq_max, pk.frq)
                tk.frq_min = min(tk.frq_min, pk.frq)
                
                # rolling averages                     
                alpha = 1 / tk.duration
                tk.frq += pk.frq * alpha
                tk.amp += pk.amp * alpha

    # process tracks again for amp & freq thresholds
    for tk in tracks:
        if tk.amp_max < amp_threshold or tk.frq_max > highest_frequency or tk.frq_min < lowest_frequency:
            tracks_for_removal.add(tk.track)
    
    
    renumbering_tracks = [None] * len(tracks)
    
    # prune invalid tracks        
    tracks = [tk for tk in tracks if tk.track not in tracks_for_removal]
    
    # sort tracks by average freq and build renumbering map and renumber tracks
    tracks.sort(key=lambda tk: tk.frq)
    for ind, tk in enumerate(tracks):
        renumbering_tracks[tk.track] = ind
        tk.track = ind
    
    # renumber and prune peaks
    for frame_n in range(len(analysis_frames)):
        new_frame = []
        for pk in analysis_frames[frame_n]:
            if renumbering_tracks[pk.track] is not None:
                pk.track = renumbering_tracks[pk.track]
                new_frame.append(pk)
        analysis_frames[frame_n] = new_frame
    
    return tracks

