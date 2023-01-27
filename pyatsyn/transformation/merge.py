# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyatsyn Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
# <Oscar Pablo Di Liscia, Pete Moss, and Juan Pampin>

"""
TODO
"""

ATS_VALID_MERGE_MATCH_MODES = [ "plain",
                                "stable",
                                "spread",
                                "full",
                                "lower",
                                "middle",
                                "higher",
                                "twist",
                                "random",
                                "randomduped"
                                ]

def merge(  ats_snd1, 
            ats_snd2,
            merge_start = 0.0,
            merge_dur = None,
            ats_snd1_start = 0.0,
            ats_snd2_start = 0.0,
            ats_snd2_dur = None,           
            match_mode = "stable",
            force_matches = None,
            time_deviation = None, # if None will us 1/ATS_DEFAULT_SAMPLING_RATE to account for floating point error
            frequency_deviation = 0.1,
            frequency_bias_curve = None,
            amplitude_bias_curve = None,
            noise_bias_curve = None,
            return_match_list_only = False,
            ):
    """
    TODO

                t=0.0 (relative to ats_snd1 0.0 + ats_snd1_start)
                 |  (e.g., ats_snd1_start = '**')
    ats_snd1  |**-------------------------------| any ats_snd1 frames after the merge range will be omitted
                            merge_start is relative to t=0.0
                            |
                            |xxxx| by default merge dur 'xxxx' will go to the end of ats_snd1

                            ats_snd2 will be aligned to merge_start offset by ats_snd2_start (e.g, ats_snd2_start = '****')
                            |
    ats_snd2            |****-------**********|
                                    |
                                    relative to ats_snd2_start, ats_snd2_dur specifies the end of the output (e.g., ats_snd2_dur = '-------'), if None to the end of the ats sound

    output       |11111111111mmmm222|


    special cases:
        if either sound has 'silent' areas within the merge any interpolation will substitute in amp of 0.0 and frq of the non-silent partial
        if ats_snd1 ends before the merge, all partials from ats_snd1 will be treated as silent for the gap and the merge, but will use the frqs from any matched partials in ats_snd2.
        if ats_snd1 is longer than the merge, it will be truncated after


    match_modes: 
        "plain" - just stitches together with no interpolation
        "stable" - uses frequency_deviation and stable matching to make optimal pairings
        "spread" - covers matching as evenly as possible mapping to the full frequency span of inputs to outputs
        "full" - every partial will have a match because we will duplicate & fork as needed (needs more info)
        "lower" - lowest frequencies are prioritized for matching
        "middle" - middle frequencies are prioritized for matching
        "higher" - higher frequencies are prioritized for matching
        "twist" - same as spread, but the bins of one side of the match are flipped lowest <-> highest
        "random" - random pairing
        "randomduped" - random pairing but every partial has a match because we will duplicate & fork as needed

    force_matches:
        If None, will process all partials according to match_modes
        or
        [(snd1_p#, snd2_p#), ... ] - a list of tuples of partial numbers to match from ats_snd1 to ats_snd2, pairing a partial with None is valid. Repeated partials will cause a duplicate partial to be 'born'.
            specified partial indices higher than the available partials indices will be replaced with None. Pairs of None, None will be ignored.
            
    NOTE: currently only supports equal noise bands
    
    TODO!!!!!! rethink what the index refers to in the list form?!!??!?!!!!!!!!!!!!!!!!!!!!!!!!!

    *bias_curve - used to specify how the mixture is made for the matched partials.
        
        *bias_curve = num # interpreted as constant bias at all times for all partials (e.g.,0.0 for all ats_snd1, 1.0 for all ats_snd2, 0.5 for 50% mixture)
            NOTE: all bias values must be in the range [0.0, 1.0]
        *bias_curve = None # interpreted as linear interpolation time envelope from 0 to 1 parallel for all partials
        *bias_curve = tuple (invalid)
        *bias_curve = envelope # interpreted as a time/bias envelope parallel for all partials
            all envelopes should be a list of tuple pairs of time and value: [(t0,v0),(t1,v1), ... (tn,vn)]
                if [(t,v)] this is the same as specifying bias v for the length of the merge
                envelope t's should be monotonically increasing and will be proportionally re-scaled to the length of the merge
                    e.g, [(0.2, 1.0), (0.1, 0.5)] is invalid and will raise an exception because t0 > t1
                    e.g., [(0.2, 1.0), (0.3, 0.5), (1.0, 0.0)] is valid and will be rescaled -> [(0.0, 1.0), (merge_dur * 0.125, 0.5), (merge_dur, 0.0)]

        *bias_curve = list of num, None or envelopes to specify the value for partial at corresponding index of base list
            *bias_curve indices in the base list correspond to the match indices (especially useful if you specify your matches directly, see match_modes)
            *if more biases are specified than there are partial matches, the extra will be ignored
            *if fewer biases are specified than there are partial matches, remaining partial matches will assume None, i.e., linear interpolation over the merge duration
            * an empty list will be interpreted as if the *bias_curve = None
        NOTE: when specifying a partial matched to None (i.e, unmatched pairing), None will be treated as amp 0.0 and freq of the existent partial.
            e.g., let's say the match is (4, None). This means partial #4 of ats_snd1 found no match. The interpolation range in effect looks like [4.frq -> 4.frq] and [4.amp -> 0.0]
        
        *bias_curve examples
            e.g., 0.5 all partials will be a 50% mix of their corresponding matches for the entire merge duration
            e.g., [None, 0.5, [(0.2, 0.7)]] is VALID, partial 0 will be linearly interpolated, partial 1 will be 50% mix of both for the entire merge duration, and partial 2 will by 70% from ats_snd2, 30% from ats_snd1
            e.g., [None, [(0.2, 1.0), (0.1, 0.5)], 2.0] is INVALID because partial 1
            e.g., [[(0.2, 1.0), (0.3, 0.5), (1.0, 0.0)], 1.0, None, None] is VALID
            e.g., [1, (0,1), [1], [(0,1),(1,0)]] is INVALID because although partial 0 & 3 are correct, partial 1 & 2 are incorrectly specified

    """

    # get new partials
    matches = []
    p1_remaining = set(range(ats_snd1.partials))
    p2_remaining = set(range(ats_snd1.partials)) 

    if force_matches is not None:        
        if is_valid_list_of_pairs(force_matches):
            # add the forced matches, and remove from the partial lists for match_mode processing
            for tp in force_matches:
                if tp != (None, None):
                    matches.append(tp)
                    p1_remaining = p1_remaining - {tp[0]}
                    p2_remaining = p2_remaining - {tp[1]}                                      
        else:
            raise Exception("force_matches not correctly specified")

    if match_mode not in ATS_VALID_MERGE_MATCH_MODES:
        raise Exception("specified match_mode is not supported")
    else:
        if match_mode == "plain":
            for p in p1_remaining:
                matches.append((p, None))
            for p in p2_remaining:
                matches.append((None, p))

        elif match_mode == "stable":
            pass # TODO

        elif match_mode == "spread":
            pass # TODO

        elif match_mode == "full":
            pass # TODO

        elif match_mode == "lower":
            pass # TODO

        elif match_mode == "middle":
            pass # TODO

        elif match_mode == "higher":
            pass # TODO

        elif match_mode == "twist":
            pass # TODO

        elif match_mode == "random":
            pass # TODO

        elif match_mode == "randomduped":
            pass # TODO
    
    if return_match_list_only:
        return matches, None
    

    # check bias curves

    # get new frames
    if merge_dur is None:
        pass #TODO set to end of file
    # if merge_dur == 0.0 are there special rules for dropping/merging ats_snd2 frame 0?

    
    

    # new AtsSoundVFR

    # add beginning

    # do merge

    # phase correction

    # update max/av & resort

    # add end

    return matches, None

def is_valid_list_of_pairs(check):
    """
    TODO
    TODO does this work with generators?
    """
    if isinstance(check, list):
        for tp in check:
            if not (isinstance(tp, tuple) and len(tp) == 2 and \
                (tp[0] is None or isinstance(tp[0], (int, float))) and \
                    (tp[1] is None or isinstance(tp[1], (int, float)))):
                return False
        return True
    return False

def is_valid_envelope(envelope):

    return True # TODO

def insert():
    pass# TODO

def chimerize():
    pass# TODO

def xfade():
    pass# TODO

def concat():
    pass# TODO

def splice():
    pass# TODO

