# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyatsyn Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
# <Oscar Pablo Di Liscia, Pete Moss, and Juan Pampin>

"""A set of functions to handle transformations using multiple ats sound objects

Attributes
----------
TODO
"""

import random
from heapq import heappop, heappush
from queue import SimpleQueue
from numpy import inf, zeros, copy

from pyatsyn.ats_structure import MatchCost, AtsSoundVFR
from pyatsyn.ats_utils import ATS_DEFAULT_SAMPLING_RATE


ATS_VALID_MERGE_MATCH_MODES = [ "plain",
                                "stable",
                                "spread",
                                "full",
                                "lower",
                                "middle",
                                "higher",
                                "twist",
                                "twist_full",
                                "random",
                                "random_full"
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
            snd1_frq_av_range = None,
            snd2_frq_av_range = None,
            time_deviation = None, # if None will us 1/ATS_DEFAULT_SAMPLING_RATE to account for floating point error, ignored by start/end of merge
            frequency_deviation = 0.1,
            frequency_bias_curve = None,
            amplitude_bias_curve = None,
            noise_bias_curve = None,
            return_match_list_only = False,
            ):
    """
    TODO


    cross-synthesis tool

                t=0.0 (relative to ats_snd1 0.0 + ats_snd1_start)
                 |  (e.g., ats_snd1_start = '**')
    ats_snd1  |**-------------------------------| any ats_snd1 frames after the merge range will be omitted
                            merge_start is relative to t=0.0
                            |
                            |xxxx| by default merge dur  will go to the end of ats_snd1 if merge_dur = None (in the depiction it's 'xxxx')

                            ats_snd2 will be aligned to merge_start offset by ats_snd2_start (e.g, ats_snd2_start = '****')
                            |
    ats_snd2            |****-------**********|
                                    |
                                    relative to ats_snd2_start, ats_snd2_dur specifies the end of the output (e.g., ats_snd2_dur = '-------'), if None to the end of the ats sound
                                    if the end of the merge_dur ends after the end of ats_snd2_dur, then merge_dur will specify the end

    output       |11111111111mmmm222|


    special cases:
        if either sound has 'silent' areas within the merge any interpolation will substitute in amp of 0.0 and frq of the non-silent partial
        if ats_snd1 ends before the merge, all partials from ats_snd1 will be treated as silent for the gap and the merge, but will use the frqs from any matched partials in ats_snd2.
        if ats_snd1 is longer than the merge, it will be truncated after


    match_modes: 
        "plain" - just stitches together with no interpolation
        "stable" - uses frequency_deviation and stable matching to make optimal pairings
            NOTE: (maybe TODO) currently uses frq_av to calculate costs. For tracks with wilder behavior (e.g., prior merges), maybe looking at average close to the merge_start is more suitable
        "spread" - covers matching as evenly as possible mapping to the full frequency span of inputs to outputs
        "full" - every partial will have a match because we will duplicate & fork as needed (needs more info)
        "lower" - lowest frequencies are prioritized for matching
        "middle" - middle frequencies are prioritized for matching
        "higher" - higher frequencies are prioritized for matching
        "twist" - same as spread, but the bins of one side of the match are flipped lowest <-> highest
        "random" - random pairing
        "random_full" - random pairing but every partial has a match because we will duplicate & fork as needed

    snd#_frq_av_range:
        NOTE: only averages non-zero frequency values, unless all are 0.0 in that range
        If None, will use .frq_av for each partial
        If float/int will interpret that as seconds from merge_start to before the time before merge_start to use for snd1, to time after merge_start for snd2 for all partials
        If 2 float/int iterable will use that as the time time range in seconds to average over relative to the ats_snd# 0.0 (not relative to ats_snd#_start)
        For other iterables:
            if fewer than the number of partials, remaining partials will use .frq_av
            if more than the number of partials, will ignore the excess
            each iterable specifies it's corresponding partial, using None, float/int or a 2 float/int iterable as stated above

    force_matches:
        If None, will process all partials according to match_modes
        or
        [(snd1_p#, snd2_p#), ... ] - a list of tuples of partial numbers to match from ats_snd1 to ats_snd2, pairing a partial with None is valid. Repeated partials will cause a duplicate partial to be 'born'.
            specified partial indices higher than the available partials indices will be replaced with None. Pairs of None, None will be ignored.
            
    NOTE: currently only supports equal noise bands
    
    TODO!!!!!! rethink what the index refers to in the list form?!!??!?!!!!!!!!!!!!!!!!!!!!!!!!!
        first - all force_matches
        then - all subsequent matches found by the match_mode
        then - all p#, None pairs remaining
        then - all None, p# pairs remaining

    *bias_curve - used to specify how the mixture is made for the matched partials.
        
        *bias_curve = num # interpreted as constant bias at all times for all partials (e.g.,0.0 for all ats_snd1, 1.0 for all ats_snd2, 0.5 for 50% mixture)
            NOTE: all bias values must be in the range [0.0, 1.0] anything outside this range will be constrained to the range (i.e. -0.5 -> 0.0 or 7.2 -> 1.0)
        *bias_curve = None # interpreted as linear interpolation time envelope from 0 to 1 parallel for all partials
        *bias_curve = tuple (invalid)
        *bias_curve = envelope # interpreted as a time/bias envelope parallel for all partials
            all envelopes should be a list of tuple pairs of time and value: [(t0,v0),(t1,v1), ... (tn,vn)]
                if [(t,v)] this is the same as specifying bias v for the length of the merge
                envelope t's should be monotonically increasing and will be proportionally re-scaled to the length of the merge
                    e.g, [(0.2, 1.0), (0.1, 0.5)] is invalid and will raise an exception because t0 > t1
                    e.g., [(0.2, 1.0), (0.3, 0.5), (1.0, -0.1)] is valid and will be rescaled -> [(0.0, 1.0), (merge_dur * 0.125, 0.5), (merge_dur, 0.0)]
                        NOTE: the final bias value was capped to 0.0. Times get rescaled, biases get constrainted.

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
    if time_deviation is None:
        time_deviation = 1 / ATS_DEFAULT_SAMPLING_RATE

    if merge_dur is None:
        merge_dur = ats_snd1.dur - ats_snd1_start
    if merge_dur < 0.0:
        merge_dur = 0.0
    merge_end = merge_start + merge_dur
    out_dur = merge_start + max(merge_dur, ats_snd2_dur)
    
    new_frame_time_candidates = [0.0]
    new_frame_time_candidates += [ats_snd1.time >= ats_snd1_start and ats_snd1.time <= ats_snd1_start + merge_end] - ats_snd1_start
    snd2_time_offset = merge_start - ats_snd2_start
    new_frame_time_candidates += [ats_snd2.time >= ats_snd2_start and ats_snd2.time <= ats_snd2_start + ats_snd2_dur] - snd2_time_offset

    # get new partials
    matches = []
    p1_remaining = set(range(ats_snd1.partials))
    p2_remaining = set(range(ats_snd1.partials)) 

    if force_matches is not None:
        check_valid, force_matches = is_valid_list_of_pairs(force_matches)
        if check_valid:
            # add the forced matches, and remove from the partial lists for match_mode processing
            for tp in force_matches:
                if tp != (None, None):
                    matches.append(tp)
                    p1_remaining = p1_remaining - {tp[0]}
                    p2_remaining = p2_remaining - {tp[1]}                                      
        else:
            raise Exception("force_matches not properly specified")

    if match_mode not in ATS_VALID_MERGE_MATCH_MODES:
        raise Exception("specified match_mode is not supported")
    
    elif match_mode == "plain":
        pass

    elif match_mode == "stable":
        p1_list = list(p1_remaining)
        p2_list = list(p2_remaining)

        p1_costs = [[] for _ in p1_remaining]

        # set frequencies to use for costs
        check_valid, snd1_cost_frq = is_valid_cost_range(snd1_frq_av_range, ats_snd1, ats_snd1_start + merge_start)
        if not check_valid:
            raise Exception("snd1_frq_av_range is not properly specified")      
        check_valid, snd2_cost_frq = is_valid_cost_range(snd2_frq_av_range, ats_snd2, ats_snd2_start, before_merge = False)
        if not check_valid:
            raise Exception("snd2_frq_av_range is not properly specified")

        # calculate costs
        for p2_ind, p2 in enumerate(p2_remaining):
            for p1_ind, p1 in (p1_remaining):
                if are_valid_frq_candidates(snd1_cost_frq[p1],snd2_cost_frq[p2],frequency_deviation):
                    cost = abs(snd1_cost_frq[p1] - snd2_cost_frq[p2])
                    heappush(p1_costs[p1_ind], MatchCost(cost, p2_ind))

        # perform Gale-Shapley stable matching
        p1_queue = SimpleQueue()
        for ind in range(len(p1_list)):
            p1_queue.put(ind)

        unmatched_p1 = []
        p2_matches = [None for _ in p2_list]

        while(not p1_queue.empty()):
            p1_ind = p1_queue.get()

            made_match = False
            while( len(p1_costs[p1_ind]) > 0):
                p2 = heappop(p1_costs[p1_ind])
                # if p2 is unmatched
                if p2_matches[p2.index] is None:
                    # update match
                    p2_matches[p2.index] = MatchCost(p2.cost, p1_ind)
                    made_match = True
                    break
                # prior match is more costly, re-match
                elif p2_matches[p2.index].cost > p2.cost:
                    # kick previous match back onto queue
                    p1_queue.put(p2_matches[p2.index].index)
                    # update match
                    p2_matches[p2.index] = MatchCost(p2.cost, p1_ind)
                    made_match = True
                    break
            # if we ran out of candidates or no match was made, the peak is unmatched
            if not made_match:
                unmatched_p1.append(p1_list[p1_ind])

        # process matches
        unmatched_p2 = []
        for p2_ind, p2 in enumerate(p2_matches):
            if p2 is None:
                unmatched_p2.append(p2_list[p2_ind])
            else:
                p1_val = p1_list[p2.index]
                matches.append((p1_val, p2_list[p2_ind]))

        # update _remaining sets
        p1_remaining = set(unmatched_p1)
        p2_remaining = set(unmatched_p2)

    else:
        p1_len = len(p1_remaining)
        p2_len = len(p2_remaining)
        
        if match_mode == "spread":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                skip = set(random.sample(list(p1_remaining), p1_len - p2_len))
                matches += zip(p1_remaining - skip, p2_remaining)
                p1_remaining = skip
                p2_remaining = {}
            else:         
                skip = set(random.sample(list(p2_remaining), p2_len - p1_len))
                matches += zip(p1_remaining, p2_remaining - skip)
                p1_remaining = {}
                p2_remaining = skip
            
        elif match_mode == "full":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
            elif p1_len > p2_len:
                dupes = set(random.sample(list(p2_remaining), p1_len - p2_len))
                matches += zip(p1_remaining, sorted(list(p2_remaining) + dupes))
            else:                         
                dupes = set(random.sample(list(p1_remaining), p2_len - p1_len))
                matches += zip(sorted(list(p1_remaining) + dupes), p2_remaining)      
            p1_remaining = {}
            p2_remaining = {}

        elif match_mode == "lower":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                matches += zip(list(p1_remaining)[:p2_len], p2_remaining)
                p1_remaining = set(list(p1_remaining)[p2_len:])
                p2_remaining = {}
            else:         
                matches += zip(p1_remaining, list(p2_remaining)[:p1_len])
                p1_remaining = {}
                p2_remaining = set(list(p2_remaining)[p1_len:])

        elif match_mode == "middle":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                mid = p1_len // 2
                lo = mid - (p2_len // 2)
                hi = lo + p2_len
                p1_list = list(p1_remaining)
                matches += zip(p1_list[lo:hi], p2_remaining)
                p1_remaining = set(p1_list[:lo] + p1_list[hi:])
                p2_remaining = {}
            else:                         
                mid = p2_len // 2
                lo = mid - (p1_len // 2)
                hi = lo + p1_len
                p2_list = list(p2_remaining)
                matches += zip(p1_remaining, p2_list[lo:hi])
                p1_remaining = {}
                p2_remaining = set(p2_list[:lo] + p2_list[hi:])

        elif match_mode == "higher":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                matches += zip(list(p1_remaining)[p1_len - p2_len:],p2_remaining)
                p1_remaining = set(list(p1_remaining)[:p1_len - p2_len])
                p2_remaining = {}
            else:         
                matches += zip(p1_remaining, list(p2_remaining)[p2_len - p1_len:])
                p1_remaining = {}
                p2_remaining = set(list(p2_remaining)[:p2_len - p1_len])

        elif match_mode == "twist":
            if p1_len == p2_len:
                matches += zip(p1_remaining, list(p2_remaining)[::-1])
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                skip = set(random.sample(list(p1_remaining), p1_len - p2_len))
                matches += zip(p1_remaining - skip, list(p2_remaining)[::-1])
                p1_remaining = skip
                p2_remaining = {}
            else:         
                skip = set(random.sample(list(p2_remaining), p2_len - p1_len))
                matches += zip(p1_remaining, list(p2_remaining - skip)[::-1])
                p1_remaining = {}
                p2_remaining = skip

        elif match_mode == "twist_full":
            if p1_len == p2_len:
                matches += zip(p1_remaining, list(p2_remaining)[::-1])
            elif p1_len > p2_len:
                dupes = set(random.sample(list(p2_remaining), p1_len - p2_len))
                matches += zip(p1_remaining, sorted(list(p2_remaining) + dupes)[::-1])
            else:                         
                dupes = set(random.sample(list(p1_remaining), p2_len - p1_len))
                matches += zip(sorted(list(p1_remaining) + dupes), list(p2_remaining)[::-1])
            p1_remaining = {}
            p2_remaining = {}

        elif match_mode == "random":
            if p1_len == p2_len:
                matches += zip(p1_remaining, random.sample(list(p2_remaining),p2_len))
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                skip = set(random.sample(list(p1_remaining), p1_len - p2_len))
                matches += zip(p1_remaining - skip, random.sample(p2_remaining, p2_len))
                p1_remaining = skip
                p2_remaining = {}
            else:         
                skip = set(random.sample(list(p2_remaining), p2_len - p1_len))
                matches += zip(p1_remaining, random.sample(p2_remaining - skip, p1_len))
                p1_remaining = {}
                p2_remaining = skip

        elif match_mode == "random_full":
            if p1_len == p2_len:
                matches += zip(p1_remaining, random.sample(list(p2_remaining),p2_len))
                p1_remaining = {}
                p2_remaining = {}
            elif p1_len > p2_len:
                dupes = set(random.sample(list(p2_remaining), p1_len - p2_len))
                matches += zip(p1_remaining, random.sample(sorted(list(p2_remaining) + dupes), p1_len))
            else:                         
                dupes = set(random.sample(list(p1_remaining), p2_len - p1_len))
                matches += zip(random.sample(sorted(list(p1_remaining) + dupes), p2_len), p2_remaining)
            p1_remaining = {}
            p2_remaining = {}
    
    # assign remaining partials to None
    for p in p1_remaining:
        matches.append((p, None))
    for p in p2_remaining:
        matches.append((None, p))
    
    if return_match_list_only:
        return matches, None

    partials = len(matches)

    # validate and build bias curves
    check_valid, env_frames, frequency_bias_curve = is_valid_bias_curve(frequency_bias_curve, merge_dur, partials, merge_start)
    if not check_valid:
        raise Exception("frequency_bias_curve not properly specified")
    new_frame_time_candidates += env_frames

    check_valid, env_frames, amplitude_bias_curve = is_valid_bias_curve(amplitude_bias_curve, merge_dur, partials, merge_start)
    if not check_valid:
        raise Exception("amplitude_bias_curve not properly specified")
    new_frame_time_candidates += env_frames
        
    new_frame_time_candidates = sorted(set(new_frame_time_candidates))
    # TODO remove based on time-deviations
    # TODO insert merge_start & merge_end
    frames = len(new_frame_time_candidates)

    # new AtsSoundVFR
    ats_out = AtsSoundVFR(frames=frames, partials=partials, dur=out_dur, has_phase=True)

    # do merge TODO
    # add beginning TODO
    # TODO what to do about duplicated partials!?
    # if merge_dur == 0.0 are there special rules for dropping/merging ats_snd2 frame 0? TODO
    # add end TODO

    # phase correction TODO

    # update max/av & resort
    ats_out.optimize()

    return matches, ats_out


def is_iterable(check):
    """
    TODO
    """
    try:
        iter(check)
    except Exception:
        return False
    else:
        return True

def is_valid_list_of_pairs(check):
    """
    TODO
    TODO does this work with generators?
    """
    out = []
    if is_iterable(check):
        for tp in check:
            if not is_iterable(tp):
                return False, None
            out.append(tuple(tp))
            if not (len(out[-1] == 2) and is_num_or_none(out[-1][0]) \
                    and is_num_or_none(out[-1][1])):                
                return False, None
        return True, out
    return False, None

def is_valid_cost_range(check, ats_snd, merge_time, before_merge = True):    
    """ TODO
    snd#_frq_av_range:
        NOTE: only averages non-zero frequency values, unless all are 0.0 in that range
        If None, will use .frq_av for each partial
        If float/int will interpret that as seconds from merge_start to before the time before merge_start to use for snd1, to time after merge_start for snd2 for all partials
        If 2 float/int iterable will use that as the time range in seconds to average over for all partials
        For other iterables:
            if fewer than the number of partials, remaining partials will use .frq_av
            if more than the number of partials, will ignore the excess
            each iterable specifies it's corresponding partial, using None, float/int or a 2 float/int iterable as stated above
    """    
    if check is None:
        return True, copy(ats_snd.frq_av)
    elif isinstance(check, (float, int)):
        out = zeros(ats_snd.partials, "float64")
        # TODO seconds before/after merge to average over for all partials, remember to check >=0.0?
        return True, out
    elif not is_iterable(check):
        return False, None
    else:
        out = zeros(ats_snd.partials, "float64")
        collect = [ck for ck in check]
        if len(collect) == 2:
            if isinstance(collect[0], (float, int)) and isinstance(collect[1], (float, int)):                
                # TODO use this as the time range to average over for all partials, remember to check the time range?
                return True, out
        for ind, ck in enumerate(collect[:ats_snd.partials]):
            if ck is None:
                out[ind] = ats_snd.frq_av[ind]
            elif isinstance(ck, (float, int)):
                pass # TODO seconds before/after merge to average over for this partial,, remember to check >=0.0?
            elif not is_iterable(ck):
                return False, None
            else:
                pass # TODO check for size 2 tuple to specify range for this partial, , remember to check the time range?
        # do remaining partials
        for ind in range(len(collect), ats_snd.partials):
            out[ind] = ats_snd.frq_av[ind]
        return True, out


def is_valid_bias_curve(check, end_time, partials, out_frame_time_offset):
    """ TODO
    *bias_curve - used to specify how the mixture is made for the matched partials.
        
        *bias_curve = num # interpreted as constant bias at all times for all partials (e.g.,0.0 for all ats_snd1, 1.0 for all ats_snd2, 0.5 for 50% mixture)
            NOTE: all bias values must be in the range [0.0, 1.0] anything outside this range will be constrained to the range (i.e. -0.5 -> 0.0 or 7.2 -> 1.0)
        *bias_curve = None # interpreted as linear interpolation time envelope from 0 to 1 parallel for all partials
        *bias_curve = tuple (invalid)
        *bias_curve = envelope # interpreted as a time/bias envelope parallel for all partials
            all envelopes should be a list of tuple pairs of time and value: [(t0,v0),(t1,v1), ... (tn,vn)]
                if [(t,v)] this is the same as specifying bias v for the length of the merge
                envelope t's should be monotonically increasing and will be proportionally re-scaled to the length of the merge
                    e.g, [(0.2, 1.0), (0.1, 0.5)] is invalid and will raise an exception because t0 > t1
                    e.g., [(0.2, 1.0), (0.3, 0.5), (1.0, -0.1)] is valid and will be rescaled -> [(0.0, 1.0), (merge_dur * 0.125, 0.5), (merge_dur, 0.0)]
                        NOTE: the final bias value was capped to 0.0. Times get rescaled, biases get constrainted.

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
    default_end_time = end_time
    if default_end_time == 0.0:
        default_end_time = 1.0
    out_env = []
    out_frames = []
    if check is None:
        out_env = [[(0.0, 0.0), (default_end_time, 1.0)] for p in partials]
        return True, [], out_env
    elif isinstance(check, (float, int)):
        check = min(max(check, 0.0), 1.0)
        out_env = [[(0.0, check), (default_end_time, check)] for p in partials]
        return True, [], out_env
    elif not is_iterable(check):
        return False, [], None
    else:
        for ck in check:
            if ck is None:
                out_env.append([[(0.0, 0.0, (default_end_time, 1.0))]])
            elif isinstance(check, (float, int)):
                outval = min(max(check, 0.0), 1.0)
                out_env.append([[(0.0, outval), (default_end_time, outval)]])
            elif not is_iterable(ck):
                return False, [], None
            else:
                # potentially a time/val envelope
                env = []
                cur_time = -inf
                for tp in ck:
                    # we expect tuples
                    if not is_iterable(tp):
                        return False, [], None
                    outval = tuple(tp)
                    # we expect 2 numbers: time, bias & time must monotonically increase
                    if len(outval) != 2 or not isinstance(outval[0], (float, int)) \
                                or not isinstance(outval[1], (float, int)) \
                                    or outval[0] <= cur_time:
                        return False, [], None
                    cur_time = outval[0]
                    outval[1] = min(max(outval[1], 0.0), 1.0)
                    env.append(outval)
                env_len = len(env)
                if env_len == 0:
                    # an empty list will be interpreted as None
                    out_env.append([[(0.0, 0.0, (default_end_time, 1.0))]])
                elif env_len == 1:
                    # a single tuple will be interpreted as that bias for the entire length of the merge
                    out_env.append([[(0.0, env[0][1]), (default_end_time, env[0][1])]])
                else:                    
                    min_time = env[0][0]
                    max_time = env[-1][0]
                    time_norm = default_end_time / (max_time - min_time)
                    env[0][0] = 0.0
                    env[-1][0] = default_end_time
                    for ind in range(1, env_len - 1):
                        # normalize times
                        env[ind][0] = time_norm * (env[ind][0] - min_time)
                        # export to out_frames
                        out_frames.append(env[ind][0] + out_frame_time_offset)
                    out_env.append(env)
        return True, out_frames, out_env

    
def is_num_or_none(check):
    """
    TODO
    """
    return check is None or isinstance(check, (int, float))

def are_valid_frq_candidates(frq1, frq2, deviation):
    """Function to determine if the distance between two frequencies are within the relative deviation constraint

    Frequencies are valid candidates for pairing if their absolute distance is smaller than the frequency deviation 
    multiplied by the lower of the candidate frequencies.

    Parameters
    ----------
    frq1 : float
        a candidate frequency
    frq2 : float
        a candidate frequency
    deviation : float
        relative frequency deviation

    Returns
    -------
    bool
        True if the candidates are within constrained range, False otherwise.
    """ 
    min_frq = min(frq1, frq2)
    return abs(frq1 - frq2) <= 0.5 * min_frq * deviation



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

def merge_CLI():
    pass# TODO