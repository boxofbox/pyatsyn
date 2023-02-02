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
from numpy import inf, zeros, copy, where, asarray
from collections import defaultdict

from pyatsyn.ats_structure import MatchCost, AtsSoundVFR
from pyatsyn.ats_utils import ATS_DEFAULT_SAMPLING_RATE, phase_interp_cubic, phase_interp_linear
from pyatsyn.analysis.critical_bands import ATS_CRITICAL_BAND_EDGES

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
                                "random_full",
                                "closest",
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
            drop_unmatched = False,
            drop_unmatched_during_merge = False,
            snd1_frq_av_range = None,
            snd2_frq_av_range = None,
            time_deviation = None, # if None will us 1/ATS_DEFAULT_SAMPLING_RATE to account for floating point error, ignored by start/end of merge
            frequency_deviation = 0.1,
            frequency_bias_curve = None,
            amplitude_bias_curve = None,
            noise_bias_curve = None, # NOTE: currently ignores .energy
            return_match_list_only = False,
            verbose = False,
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
        "closest" - closest frequency, ignoring frequency deviation, allows dupes

    snd#_frq_av_range:
        NOTE: only averages non-zero frequency values, unless all are 0.0 in that range
        If None, will use .frq_av for each partial
        If float/int will interpret that as seconds from merge_start to before the time before merge_start to use for snd1, to time after merge_start for snd2 for all partials, must be > 0.0
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
            
            TODO: change envelope docs to [] of []s not tuples

    """
    if verbose:
        print("Beginning merge pre-processing...")

    if time_deviation is None:
        time_deviation = 1 / ATS_DEFAULT_SAMPLING_RATE

    ats_snd1_start = max(0.0, ats_snd1_start)
    ats_snd2_start = max(0.0, ats_snd2_start)

    if ats_snd1_start >= ats_snd1.dur:
        raise Exception("TODO")
    if ats_snd2_start >= ats_snd2.dur:
        raise Exception("TODO")

    merge_start = max(0.0, merge_start)

    if merge_dur is None:
        merge_dur = ats_snd1.dur - (ats_snd1_start + merge_start)
    merge_dur = max(0.0, merge_dur)    
    merge_end = merge_start + merge_dur

    if ats_snd2_dur is None:
        ats_snd2_dur = ats_snd2.dur - ats_snd2_start
    ats_snd2_dur = max(0.0, ats_snd2_dur)

    out_dur = merge_start + max(merge_dur, ats_snd2_dur)
    
    new_frame_time_candidates = [0.0]    
    new_frame_time_candidates += list(ats_snd1.time[(ats_snd1.time >= ats_snd1_start) & (ats_snd1.time <= ats_snd1_start + merge_end)] - ats_snd1_start)
    snd2_time_offset = merge_start - ats_snd2_start
    new_frame_time_candidates += list(ats_snd2.time[(ats_snd2.time >= ats_snd2_start) & (ats_snd2.time <= ats_snd2_start + ats_snd2_dur)] + snd2_time_offset)

    ##################
    # MATCH PARTIALS #
    ##################

    if verbose:
        print("Matching partials...")

    # get new partials
    matches = []
    p1_remaining = set(range(ats_snd1.partials))
    p2_remaining = set(range(ats_snd2.partials)) 

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
            for p1_ind, p1 in enumerate(p1_remaining):
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
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                skip = set(random.sample(list(p1_remaining), p1_len - p2_len))
                matches += zip(p1_remaining - skip, p2_remaining)
                p1_remaining = skip
                p2_remaining = set()
            else:         
                skip = set(random.sample(list(p2_remaining), p2_len - p1_len))
                matches += zip(p1_remaining, p2_remaining - skip)
                p1_remaining = set()
                p2_remaining = skip
            
        elif match_mode == "full":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
            elif p1_len > p2_len:
                dupes = random.choices(list(p2_remaining), k = p1_len - p2_len)
                matches += zip(p1_remaining, sorted(list(p2_remaining) + dupes))
            else:                         
                dupes = random.choices(list(p1_remaining), k = p2_len - p1_len)
                matches += zip(sorted(list(p1_remaining) + dupes), p2_remaining)      
            p1_remaining = set()
            p2_remaining = set()

        elif match_mode == "lower":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                matches += zip(list(p1_remaining)[:p2_len], p2_remaining)
                p1_remaining = set(list(p1_remaining)[p2_len:])
                p2_remaining = set()
            else:         
                matches += zip(p1_remaining, list(p2_remaining)[:p1_len])
                p1_remaining = set()
                p2_remaining = set(list(p2_remaining)[p1_len:])

        elif match_mode == "middle":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                mid = p1_len // 2
                lo = mid - (p2_len // 2)
                hi = lo + p2_len
                p1_list = list(p1_remaining)
                matches += zip(p1_list[lo:hi], p2_remaining)
                p1_remaining = set(p1_list[:lo] + p1_list[hi:])
                p2_remaining = set()
            else:                         
                mid = p2_len // 2
                lo = mid - (p1_len // 2)
                hi = lo + p1_len
                p2_list = list(p2_remaining)
                matches += zip(p1_remaining, p2_list[lo:hi])
                p1_remaining = set()
                p2_remaining = set(p2_list[:lo] + p2_list[hi:])

        elif match_mode == "higher":
            if p1_len == p2_len:
                matches += zip(p1_remaining, p2_remaining)
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                matches += zip(list(p1_remaining)[p1_len - p2_len:],p2_remaining)
                p1_remaining = set(list(p1_remaining)[:p1_len - p2_len])
                p2_remaining = set()
            else:         
                matches += zip(p1_remaining, list(p2_remaining)[p2_len - p1_len:])
                p1_remaining = set()
                p2_remaining = set(list(p2_remaining)[:p2_len - p1_len])

        elif match_mode == "twist":
            if p1_len == p2_len:
                matches += zip(p1_remaining, list(p2_remaining)[::-1])
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                skip = set(random.sample(list(p1_remaining), p1_len - p2_len))
                matches += zip(p1_remaining - skip, list(p2_remaining)[::-1])
                p1_remaining = skip
                p2_remaining = set()
            else:         
                skip = set(random.sample(list(p2_remaining), p2_len - p1_len))
                matches += zip(p1_remaining, list(p2_remaining - skip)[::-1])
                p1_remaining = set()
                p2_remaining = skip

        elif match_mode == "twist_full":
            if p1_len == p2_len:
                matches += zip(p1_remaining, list(p2_remaining)[::-1])
            elif p1_len > p2_len:
                dupes = random.choices(list(p2_remaining), k = p1_len - p2_len)
                matches += zip(p1_remaining, sorted(list(p2_remaining) + dupes)[::-1])
            else:                         
                dupes = random.choices(list(p1_remaining), k = p2_len - p1_len)
                matches += zip(sorted(list(p1_remaining) + dupes), list(p2_remaining)[::-1])
            p1_remaining = set()
            p2_remaining = set()

        elif match_mode == "random":
            if p1_len == p2_len:
                matches += zip(p1_remaining, random.sample(list(p2_remaining),p2_len))
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                skip = set(random.sample(list(p1_remaining), p1_len - p2_len))
                matches += zip(p1_remaining - skip, random.sample(list(p2_remaining), p2_len))
                p1_remaining = skip
                p2_remaining = set()
            else:         
                skip = set(random.sample(list(p2_remaining), p2_len - p1_len))
                matches += zip(p1_remaining, random.sample(list(p2_remaining - skip), p1_len))
                p1_remaining = set()
                p2_remaining = skip

        elif match_mode == "random_full":
            if p1_len == p2_len:
                matches += zip(p1_remaining, random.sample(list(p2_remaining),p2_len))
                p1_remaining = set()
                p2_remaining = set()
            elif p1_len > p2_len:
                dupes = random.choices(list(p2_remaining), k = p1_len - p2_len)
                matches += zip(p1_remaining, random.sample(sorted(list(p2_remaining) + dupes), p1_len))
            else:                         
                dupes = random.choices(list(p1_remaining), k = p2_len - p1_len)
                matches += zip(random.sample(sorted(list(p1_remaining) + dupes), p2_len), p2_remaining)
            p1_remaining = set()
            p2_remaining = set()

        elif match_mode == "closest":

            # set frequencies to use for costs
            check_valid, snd1_cost_frq = is_valid_cost_range(snd1_frq_av_range, ats_snd1, ats_snd1_start + merge_start)
            if not check_valid:
                raise Exception("snd1_frq_av_range is not properly specified")      
            check_valid, snd2_cost_frq = is_valid_cost_range(snd2_frq_av_range, ats_snd2, ats_snd2_start, before_merge = False)
            if not check_valid:
                raise Exception("snd2_frq_av_range is not properly specified")

            p1_costs = {p1:inf for p1 in p1_remaining}
            p1_matches = {p1:-1 for p1 in p1_remaining}  
            p2_costs = {p2:inf for p2 in p2_remaining}
            p2_matches = {p2:-1 for p2 in p2_remaining}
            for p1 in p1_remaining:
                for p2 in p2_remaining:
                    cost = abs(snd1_cost_frq[p1] - snd2_cost_frq[p2])
                    if p1_costs[p1] > cost:
                        p1_costs[p1] = cost
                        p1_matches[p1] = p2
                    if p2_costs[p2] > cost:
                        p2_costs[p2] = cost
                        p2_matches[p2] = p1
            out_matches = set()
            for p1, p2 in p1_matches.items():
                out_matches.add((p1,p2))
            for p2, p1 in p2_matches.items():
                out_matches.add((p1,p2))
            matches += list(out_matches)
            p1_remaining = set()
            p2_remaining = set()
                    
    
    # assign remaining partials to None
    if not drop_unmatched:
        for p in p1_remaining:
            matches.append((p, None))
        for p in p2_remaining:
            matches.append((None, p))
        
    if return_match_list_only:
        return matches, None

    partials = len(matches)

    if verbose:
        print(f"{partials} partial pair(s) mapped")

    # assign partial tracks
    snd1_partial_map = defaultdict(list)
    snd2_partial_map = defaultdict(list)
    for ind, match in enumerate(matches):
        if (match[0] is not None and match[0] >= ats_snd1.partials) or (match[1] is not None and match[1] >= ats_snd2.partials):
            print("WARNING: skipping match with partial specified outside of available partial range")
            continue
        if match[0] is not None:
            snd1_partial_map[match[0]].append(ind)
        if match[1] is not None:
            snd2_partial_map[match[1]].append(ind)

    #######################
    # PROCESS BIAS CURVES #
    #######################

    if verbose:
        print("Processing bias curves...")

    # validate and build bias curves
    check_valid, env_frames, frequency_bias_curve = is_valid_bias_curve(frequency_bias_curve, merge_dur, partials, merge_start)
    if not check_valid:
        raise Exception("frequency_bias_curve not properly specified")
    new_frame_time_candidates = new_frame_time_candidates + env_frames

    check_valid, env_frames, amplitude_bias_curve = is_valid_bias_curve(amplitude_bias_curve, merge_dur, partials, merge_start)
    if not check_valid:
        raise Exception("amplitude_bias_curve not properly specified")
    new_frame_time_candidates = new_frame_time_candidates + env_frames
    
    has_noi = len(ats_snd1.bands) > 0 and len(ats_snd1.band_energy) > 0 and len(ats_snd2.bands) > 0 and len(ats_snd2.band_energy) > 0
    if has_noi:
        check_valid, env_frames, noise_bias_curve = is_valid_bias_curve(noise_bias_curve, merge_dur, len(ats_snd1.bands), merge_start)
        if not check_valid:
            raise Exception("noise_bias_curve not properly specified")
        new_frame_time_candidates = new_frame_time_candidates + env_frames
    
    #######################
    # PROCESS FRAME TIMES #
    #######################

    if verbose:
        print("Processing frame times...")    

    # make sure new time frames are not within time_deviation of each other
    new_frame_time_candidates = sorted(set(new_frame_time_candidates))  
    ind = 1
    candidates_len = len(new_frame_time_candidates)
    new_frames = None
    pad = 1 / ATS_DEFAULT_SAMPLING_RATE / 10 # pad for floating point error
    if candidates_len == 0:
        new_frames = sorted(list({0.0, merge_start, merge_end}))
        merge_start_ind = new_frames.index(merge_start)
        merge_end_ind = new_frames.index(merge_end)
    else:
        check = time_deviation
        if new_frame_time_candidates[0] != 0.0:
            new_frames = [0.0]
            ind = 0
        else:
            new_frames = [new_frame_time_candidates[0]]
        while (ind < candidates_len):
            if new_frame_time_candidates[ind] > check:
                check = new_frame_time_candidates[ind] + time_deviation
                new_frames.append(new_frame_time_candidates[ind])        
            ind += 1
        
        # insert merge_start & merge_end using 1/sampling_rate (pad) as time_deviation for floating point error
        merge_start_ind, new_frames = insert_into_list_with_deviation(new_frames, merge_start, pad, start_at = 0)
        merge_end_ind, new_frames = insert_into_list_with_deviation(new_frames, merge_end, pad, start_at=merge_start_ind)

    # force first frame to 0.0 (corrects for padding error if merge_start very close to 0)
    new_frames[0] = 0.0

    # add extra frame beyond dur for synth interpolation (also accounts for merge_end very close to out_dur, within padding error)
    if new_frames[-1] < out_dur:
        if new_frames[-1] + pad >= out_dur:
            new_frames[-1] = out_dur
        else:
            new_frames.append(out_dur)
        new_frames.append(out_dur + time_deviation)
    elif new_frames[-1] == out_dur:      
        new_frames.append(out_dur + time_deviation)

    frames = len(new_frames)

    if verbose:
        print(f"{frames} frame(s) will be used")
        print("Initializing AtsSoundVFR...")
    
    ####################
    # INIT ATSSOUNDVFR #
    ####################

    # new AtsSoundVFR
    has_pha = ats_snd1.pha is not None and ats_snd2.pha is not None
    ats_out = AtsSoundVFR(frames=frames, partials=partials, dur=out_dur, has_phase=True)

    ats_out.time = asarray(new_frames, dtype = "float64")

    if has_noi:
        ats_out.bands = copy(ats_snd1.bands)
        ats_out.band_energy = zeros([len(ATS_CRITICAL_BAND_EDGES) - 1, frames], "float64")

    last_frame_ind = ats_out.frames - 1

    ###############################
    # GET IMPORTANT MERGE INDICES #
    ###############################

    if verbose:
        print("Computing merge indices...")

    # handle start point of snd1
    snd1_start_frame = 0
    while (snd1_start_frame < ats_snd1.frames - 1):
        if within_dev(ats_snd1.time[snd1_start_frame], ats_snd1_start, pad) \
                or ats_snd1.time[snd1_start_frame] > ats_snd1_start:
            break
        snd1_start_frame += 1
        
    # handle merge start point of snd1
    snd1_end_frame = None
    snd1_start_merge_frame = snd1_start_frame
    snd1_early_cutoff_ind = None
    snd1_dur_in_out_time = ats_snd1.dur - ats_snd1_start

    merge_start_in_snd1_time = merge_start + ats_snd1_start    
    if merge_start_in_snd1_time - pad > ats_snd1.dur:
        # the merge starts after we run out of ats1_snd        
        snd1_start_merge_frame = None
        snd1_end_frame = last_frame_ind
        snd1_early_cutoff_out_ind = 0        
        while (snd1_early_cutoff_out_ind < last_frame_ind):
            if within_dev(ats_out.time[snd1_early_cutoff_out_ind], snd1_dur_in_out_time, pad) \
                    or ats_out.time[snd1_early_cutoff_out_ind] > snd1_dur_in_out_time:
                break
            snd1_early_cutoff_ind += 1
    else:        
        while (snd1_start_merge_frame < last_frame_ind):      
            if within_dev(ats_snd1.time[snd1_start_merge_frame], merge_start_in_snd1_time, pad) \
                    or ats_snd1.time[snd1_start_merge_frame] > merge_start_in_snd1_time:
                break
            snd1_start_merge_frame += 1
                        
    # handle merge end point of snd1
    merge_end_in_snd1_time = merge_end + ats_snd1_start    
    if snd1_early_cutoff_ind is None:
        if merge_end_in_snd1_time - pad > ats_snd1.dur:
            # the merge ends after we run out of ats1_snd
            snd1_end_frame = last_frame_ind
            snd1_early_cutoff_ind = merge_start_ind
            while (snd1_early_cutoff_ind < last_frame_ind):
                if within_dev(ats_out.time[snd1_early_cutoff_ind], snd1_dur_in_out_time, pad) \
                        or ats_out.time[snd1_early_cutoff_ind] > snd1_dur_in_out_time:
                    break
                snd1_early_cutoff_ind += 1
        else:
            snd1_end_frame = snd1_start_merge_frame
            while (snd1_end_frame < ats_snd1.frames - 1):
                if within_dev(ats_snd1.time[snd1_end_frame], merge_end_in_snd1_time, pad) \
                        or ats_snd1.time[snd1_end_frame] > merge_end_in_snd1_time:
                    break
                snd1_end_frame += 1
    

    # handle start point of snd2 (merge start)
    snd2_start_merge_frame = 0
    while (snd2_start_merge_frame < ats_snd2.frames - 1):
        if within_dev(ats_snd2.time[snd2_start_merge_frame], ats_snd2_start, pad):
            break
        snd2_start_merge_frame += 1

    # handle merge end point of snd2
    snd2_end_frame = None
    snd2_end_merge_frame = snd2_start_merge_frame
    snd2_early_cutoff_ind = None
    snd2_dur_in_out_time = ats_snd2.dur - ats_snd2_start + merge_start

    merge_end_in_snd2_time = merge_dur + ats_snd2_start
    if merge_end_in_snd2_time - pad > ats_snd2.dur:
        # the merge ends after we run out of ats2_snd
        snd2_end_merge_frame = None
        snd2_end_frame = ats_snd2.frames - 1
        snd2_early_cutoff_ind = merge_start_ind
        while (snd2_early_cutoff_ind < last_frame_ind):
            if within_dev(ats_out.time[snd2_early_cutoff_ind], snd2_dur_in_out_time, pad) \
                    or ats_out.time[snd2_early_cutoff_ind] > snd2_dur_in_out_time:
                break
            snd2_early_cutoff_ind += 1
    else:        
        while (snd2_end_merge_frame < ats_snd2.frames - 1):
            if within_dev(ats_snd2.time[snd2_end_merge_frame], merge_end_in_snd2_time, pad) \
                    or ats_snd2.time[snd2_end_merge_frame] > merge_end_in_snd2_time:
                break
            snd2_end_merge_frame += 1
    
    # handle end of snd2 beyond merge
    out_end_in_snd2_time = out_dur - merge_start + ats_snd2_start
    if snd2_early_cutoff_ind is None:
        if out_end_in_snd2_time - pad > ats_snd2.dur:
            # the output ends after we run out of ats2_snd
            snd2_end_frame = ats_snd2.frames - 1
            snd2_early_cutoff_ind = merge_end_ind
            while (snd2_early_cutoff_ind < last_frame_ind):
                if within_dev(ats_out.time[snd2_early_cutoff_ind], snd2_dur_in_out_time, pad) \
                        or ats_out.time[snd2_early_cutoff_ind] > snd2_dur_in_out_time:                
                    break
                snd2_early_cutoff_ind += 1
        else:
            snd2_end_frame = snd2_end_merge_frame
            while (snd2_end_frame < ats_snd2.frames - 1):
                if within_dev(ats_snd2.time[snd2_end_frame], out_end_in_snd2_time, pad) \
                        or ats_snd2.time[snd2_end_frame] > out_end_in_snd2_time:
                    break
                snd2_end_frame += 1

    #################
    # PERFORM MERGE #
    #################

    if verbose:
        print("Starting merge...")

    # add beginning
    if merge_start_ind > 0:
        if verbose:
            print("Adding pre-merge section...")
        stop = merge_start_ind
        if snd1_early_cutoff_ind is not None and snd1_early_cutoff_ind < stop:
            stop = snd1_early_cutoff_ind + 1     
        for frame_n, frame_t in enumerate(ats_out.time[:stop]):
            
            snd1_ind = snd1_start_frame
            exact = False
            frame_t_in_snd1_time = frame_t + ats_snd1_start
            while(snd1_ind < ats_snd1.frames - 1):
                if within_dev(ats_snd1.time[snd1_ind], frame_t_in_snd1_time, pad):
                    exact = True
                    break
                elif ats_snd1.time[snd1_ind] > frame_t_in_snd1_time:
                    break
                snd1_ind += 1
            
            if exact:
                # just copy the exactly timed frame over
                for pt, inds in snd1_partial_map.items():
                    ats_out.frq[inds[0]][frame_n] = ats_snd1.frq[pt][snd1_ind]
                    ats_out.amp[inds[0]][frame_n] = ats_snd1.amp[pt][snd1_ind]
                if has_pha:
                    for pt, inds in snd1_partial_map.items():
                        ats_out.pha[inds[0]][frame_n] = ats_snd1.pha[pt][snd1_ind]                        
                if has_noi:
                    ats_out.band_energy[:, frame_n] = copy(ats_snd1.band_energy[:, snd1_ind])
            else:
                # otherwise we need to interpolate from snd1
                prior_ind = snd1_ind - 1
                i_delta = frame_t_in_snd1_time - ats_snd1.time[prior_ind]
                t_delta = ats_snd1.time[snd1_ind] - ats_snd1.time[prior_ind]
                interp = i_delta / t_delta
                for pt, inds in snd1_partial_map.items():
                    ats_out.frq[inds[0]][frame_n] = ((ats_snd1.frq[pt][snd1_ind] - ats_snd1.frq[pt][prior_ind]) * interp) \
                            + ats_snd1.frq[pt][prior_ind]
                    ats_out.amp[inds[0]][frame_n] = ((ats_snd1.amp[pt][snd1_ind] - ats_snd1.amp[pt][prior_ind]) * interp) \
                            + ats_snd1.amp[pt][prior_ind]
                if has_pha:
                    for pt, inds in snd1_partial_map.items():
                        ats_out.pha[inds[0]][frame_n] = phase_interp_cubic( ats_snd1.frq[pt][prior_ind], 
                                                                            ats_snd1.frq[pt][snd1_ind],
                                                                            ats_snd1.pha[pt][prior_ind], 
                                                                            ats_snd1.pha[pt][snd1_ind],
                                                                            i_samps_from_0 = i_delta * ATS_DEFAULT_SAMPLING_RATE,
                                                                            samps_from_0_to_t= t_delta * ATS_DEFAULT_SAMPLING_RATE,
                                                                            sampling_rate=ATS_DEFAULT_SAMPLING_RATE
                                                                            )
                if has_noi:                    
                    ats_out.band_energy[:,frame_n] = ((ats_snd1.band_energy[:,snd1_ind] - ats_snd1.band_energy[:,prior_ind]) * interp) \
                           + ats_snd1.band_energy[:,prior_ind]
    
    # add merged middle
    if verbose:
        print("Merging...")

    snd1_stop = snd1_end_frame + 1
    snd2_stop = snd2_end_merge_frame
    if snd2_stop is None:
        snd2_stop = snd2_end_frame
    snd2_stop += 1

    stop = merge_end_ind + 1

    frame_n = merge_start_ind

    if merge_dur == 0.0:
        snd1_stop = merge_start_ind + 1
        snd2_stop = merge_start_ind + 1
        stop = merge_end_ind + 1
    
    snd1_ind = snd1_start_merge_frame
    snd2_ind = snd2_start_merge_frame

    while (frame_n < stop and (snd1_ind < snd1_stop or snd1_ind < snd2_stop)):
        frame_t = ats_out.time[frame_n]

        # get indices
        frame_t_in_snd1_time = frame_t + ats_snd1_start
        frame_t_in_snd2_time = frame_t + ats_snd2_start - merge_start
        frame_t_in_bias_time = frame_t - merge_start
        if merge_dur == 0.0:
            frame_t_in_bias_time = 0.5

        snd1_interp = None
        snd1_prior_ind = None
        snd2_interp = None
        snd2_prior_ind = None

        while(snd1_ind < snd1_stop):
            if within_dev(ats_snd1.time[snd1_ind], frame_t_in_snd1_time, pad):
                break
            elif ats_snd1.time[snd1_ind] > frame_t_in_snd1_time:
                snd1_prior_ind = snd1_ind - 1
                snd1_interp = (frame_t_in_snd1_time - ats_snd1.time[snd1_prior_ind]) / (ats_snd1.time[snd1_ind] - ats_snd1.time[snd1_prior_ind])
                break
            snd1_ind += 1
        while(snd2_ind < snd2_stop):
            if within_dev(ats_snd2.time[snd2_ind], frame_t_in_snd2_time, pad):
                break
            elif ats_snd2.time[snd2_ind] > frame_t_in_snd2_time:
                snd2_prior_ind = snd2_ind - 1
                snd2_interp = (frame_t_in_snd2_time - ats_snd2.time[snd2_prior_ind]) / (ats_snd2.time[snd2_ind] - ats_snd2.time[snd2_prior_ind])
                break
            snd2_ind += 1

        for ind, match in enumerate(matches):
            if (match[0] is not None and match[0] >= ats_snd1.partials) or (match[1] is not None and match[1] >= ats_snd2.partials):
                continue
            if match[0] is None and match[1] is None:
                continue
            if drop_unmatched_during_merge and (match[0] is None or match[1] is None):
                continue

            snd1_frq = 0.0
            snd1_amp = 0.0
            snd2_frq = 0.0
            snd2_amp = 0.0

            if match[0] is not None and snd1_ind < snd1_stop:
                snd1_n_bins = len(snd1_partial_map[match[0]])
                if snd1_interp is None:
                    snd1_frq = ats_snd1.frq[match[0]][snd1_ind]
                    snd1_amp = ats_snd1.amp[match[0]][snd1_ind] / snd1_n_bins
                else:                    
                    snd1_frq = ((ats_snd1.frq[match[0]][snd1_ind] - ats_snd1.frq[match[0]][snd1_prior_ind]) * snd1_interp) \
                            + ats_snd1.frq[match[0]][snd1_prior_ind]
                    snd1_amp = ((ats_snd1.amp[match[0]][snd1_ind] - ats_snd1.amp[match[0]][snd1_prior_ind]) * snd1_interp) \
                            + ats_snd1.amp[match[0]][snd1_prior_ind] / snd1_n_bins

            if match[1] is not None and snd2_ind < snd2_stop:
                snd2_n_bins = len(snd2_partial_map[match[1]])
                if snd2_interp is None:
                    snd2_frq = ats_snd2.frq[match[1]][snd2_ind]
                    snd2_amp = ats_snd2.amp[match[1]][snd2_ind] / snd2_n_bins
                else:                    
                    snd2_frq = ((ats_snd2.frq[match[1]][snd2_ind] - ats_snd2.frq[match[1]][snd2_prior_ind]) * snd2_interp) \
                            + ats_snd2.frq[match[1]][snd2_prior_ind]
                    snd2_amp = ((ats_snd2.amp[match[1]][snd2_ind] - ats_snd2.amp[match[1]][snd2_prior_ind]) * snd2_interp) \
                            + ats_snd2.amp[match[1]][snd2_prior_ind] / snd2_n_bins

            a_bias = get_env_val_at_t(amplitude_bias_curve[ind], frame_t_in_bias_time)
            
            if snd1_ind < snd1_stop and (match[1] is None or (snd2_frq == 0.0 and snd2_amp == 0.0)):
                ats_out.frq[ind][frame_n] = snd1_frq
                ats_out.amp[ind][frame_n] = snd1_amp * (1.0 - a_bias)
            elif snd2_ind < snd2_stop and (match[0] is None or (snd1_frq == 0.0 and snd1_amp == 0.0)):
                ats_out.frq[ind][frame_n] = snd2_frq
                ats_out.amp[ind][frame_n] = snd2_amp * a_bias
            elif snd1_ind < snd1_stop and snd2_ind < snd2_stop:
                f_bias = get_env_val_at_t(frequency_bias_curve[ind], frame_t_in_bias_time)
                ats_out.frq[ind][frame_n] = ((1.0 - f_bias) * snd1_frq) + (f_bias * snd2_frq)
                ats_out.amp[ind][frame_n] = ((1.0 - a_bias) * snd1_amp) + (a_bias * snd2_amp)
        
        if has_noi:
            if snd1_ind < snd1_stop:
                if snd1_interp is None:
                    for band in ats_out.bands:
                        n_bias = 1.0 - get_env_val_at_t(noise_bias_curve[band], frame_t_in_bias_time)
                        ats_out.band_energy[band][frame_n] = ats_snd1.band_energy[band][snd1_ind] * n_bias
                else:
                    for band in ats_out.bands:
                        n_bias = 1.0 - get_env_val_at_t(noise_bias_curve[band], frame_t_in_bias_time)
                        ats_out.band_energy[band][frame_n] = (((ats_snd1.band_energy[band][snd1_ind] \
                                - ats_snd1.band_energy[band][snd1_prior_ind]) * snd1_interp) \
                                    + ats_snd1.band_energy[band][snd1_prior_ind]) * n_bias

            if snd2_ind < snd2_stop:
                if snd2_interp is None:
                    for band in ats_out.bands:
                        n_bias = get_env_val_at_t(noise_bias_curve[band], frame_t_in_bias_time)
                        ats_out.band_energy[band][frame_n] += ats_snd2.band_energy[band][snd2_ind] * n_bias
                else:
                    for band in ats_out.bands:
                        n_bias = get_env_val_at_t(noise_bias_curve[band], frame_t_in_bias_time)
                        ats_out.band_energy[band][frame_n] += (((ats_snd2.band_energy[band][snd2_ind] \
                                - ats_snd2.band_energy[band][snd2_prior_ind]) * snd2_interp) \
                                    + ats_snd2.band_energy[band][snd2_prior_ind]) * n_bias

        frame_n += 1

    # add end
    if verbose:
        print("Adding post-merge section...")

    stop = ats_out.frames
    if snd2_early_cutoff_ind is not None and snd2_early_cutoff_ind < stop:
        stop = snd2_early_cutoff_ind + 1
    
    frame_n = merge_end_ind + 1
    
    while (frame_n < stop):
        frame_t = ats_out.time[frame_n]
        frame_t_in_snd2_time = frame_t + ats_snd2_start - merge_start

        snd2_ind = snd2_end_merge_frame
        exact = False
        while (snd2_ind < ats_snd2.frames - 1):
            if within_dev(ats_snd2.time[snd2_ind], frame_t_in_snd2_time, pad):
                exact = True
                break
            elif ats_snd2.time[snd2_ind] > frame_t_in_snd2_time:
                break
            snd2_ind += 1

        if exact:
            # just copy the exactly timed frame over
            for pt, inds in snd2_partial_map.items():
                ats_out.frq[inds[0]][frame_n] = ats_snd2.frq[pt][snd2_ind]
                ats_out.amp[inds[0]][frame_n] = ats_snd2.amp[pt][snd2_ind]
            if has_noi:
                ats_out.band_energy[:, frame_n] = copy(ats_snd2.band_energy[:, snd2_ind])
        
        else:
            # otherwise we need to interpolate from snd2
            prior_ind = snd2_ind - 1
            interp = (frame_t_in_snd2_time - ats_snd2.time[prior_ind]) / (ats_snd2.time[snd2_ind] - ats_snd2.time[prior_ind])
            for pt, inds in snd2_partial_map.items():
                ats_out.frq[inds[0]][frame_n] = ((ats_snd2.frq[pt][snd1_ind] - ats_snd2.frq[pt][prior_ind]) * interp) \
                        + ats_snd2.frq[pt][prior_ind]
                ats_out.amp[inds[0]][frame_n] = ((ats_snd2.amp[pt][snd1_ind] - ats_snd2.amp[pt][prior_ind]) * interp) \
                        + ats_snd2.amp[pt][prior_ind]
            if has_noi:
                ats_out.band_energy[:, frame_n] = ((ats_snd2.band_energy[:, snd2_ind] - ats_snd2.band_energy[:, prior_ind]) * interp) \
                        + ats_snd2.band_energy[:, prior_ind]
        
        frame_n += 1

    ##################
    # CLEAN-UP TASKS #
    ##################

    if verbose:
        print("Cleaning up transition points...")

     # handle 0's in prior and latter frames at transition points (remember to set freq to same +- 1 frame)     
    if merge_start_ind > 0:
        for p in range(ats_out.partials):
            if ats_out.frq[p][merge_start_ind] > 0.0 and ats_out.frq[p][merge_start_ind - 1] == 0.0 and ats_out.amp[p][merge_start_ind - 1] == 0.0:
                ats_out.frq[p][merge_start_ind - 1] == ats_out.frq[p][merge_start_ind]
    if merge_start_ind < last_frame_ind:
        for p in range(ats_out.partials):
            if ats_out.frq[p][merge_start_ind] > 0.0 and ats_out.frq[p][merge_start_ind + 1] == 0.0 and ats_out.amp[p][merge_start_ind + 1] == 0.0:
                ats_out.frq[p][merge_start_ind + 1] == ats_out.frq[p][merge_start_ind]        
    if merge_end_ind > merge_start_ind:
        for p in range(ats_out.partials):
            if ats_out.frq[p][merge_end_ind] > 0.0 and ats_out.frq[p][merge_end_ind - 1] == 0.0 and ats_out.amp[p][merge_end_ind - 1] == 0.0:
                ats_out.frq[p][merge_end_ind - 1] == ats_out.frq[p][merge_end_ind]        
        if merge_end_ind < last_frame_ind:
            for p in range(ats_out.partials):
                if ats_out.frq[p][merge_end_ind] > 0.0 and ats_out.frq[p][merge_end_ind + 1] == 0.0 and ats_out.amp[p][merge_end_ind + 1] == 0.0:
                    ats_out.frq[p][merge_end_ind + 1] == ats_out.frq[p][merge_end_ind]            
    if snd1_early_cutoff_ind is not None and snd1_early_cutoff_ind not in (merge_start_ind, merge_end_ind) \
            and snd1_early_cutoff_ind < last_frame_ind:
        for p in range(ats_out.partials):
            if ats_out.frq[p][snd1_early_cutoff_ind] > 0.0 and ats_out.frq[p][snd1_early_cutoff_ind + 1] == 0.0 and ats_out.amp[p][snd1_early_cutoff_ind + 1] == 0.0:
                ats_out.frq[p][snd1_early_cutoff_ind + 1] == ats_out.frq[p][snd1_early_cutoff_ind]
    if snd2_early_cutoff_ind is not None and snd2_early_cutoff_ind not in (merge_start_ind, merge_end_ind, snd1_early_cutoff_ind) \
            and snd2_early_cutoff_ind < last_frame_ind:
        for p in range(ats_out.partials):
            if ats_out.frq[p][snd2_early_cutoff_ind] > 0.0 and ats_out.frq[p][snd2_early_cutoff_ind + 1] == 0.0 and ats_out.amp[p][snd2_early_cutoff_ind + 1] == 0.0:
                ats_out.frq[p][snd2_early_cutoff_ind + 1] == ats_out.frq[p][snd2_early_cutoff_ind]
    if last_frame_ind > 0:
        for p in range(ats_out.partials):
            if ats_out.frq[p][last_frame_ind] > 0.0 and ats_out.frq[p][last_frame_ind - 1] == 0.0 and ats_out.amp[p][last_frame_ind - 1] == 0.0:
                ats_out.frq[p][last_frame_ind - 1] == ats_out.frq[p][last_frame_ind]        
          
    # phase correction
    if has_pha:
        if verbose:
            print("Performing phase correction...")
        for frame_n in range(max(1, merge_start_ind), ats_out.frames):
            t = ats_out.time[frame_n] - ats_out.time[frame_n - 1]
            for p in range(ats_out.partials):
                if ats_out.frq[p][frame_n] > 0.0 and ats_out.frq[p][frame_n - 1] > 0.0:
                    ats_out.pha[p][frame_n] = phase_interp_linear(ats_out.frq[p][frame_n - 1], ats_out.frq[p][frame_n], ats_out.pha[p][frame_n - 1], t)

    # update max/av & resort
    if verbose:
        print("Optimizing output...")
    ats_out.optimize()

    return matches, ats_out

def get_env_val_at_t(env, t):    
    """
    TODO
    """
    if len(env) == 0:
        return 0.0
    elif t <= env[0][0]:
        return env[0][1]
    elif t > env[-1][0]:
        return env[-1][1]
    
    for i in range(1, len(env)):
        if t == env[i][0]:
            return env[i][1]
        elif t < env[i][0]:
            interp = (t - env[i-1][0]) / (env[i][0] - env[i-1][0])
            return ((env[i][1] - env[i-1][1]) * interp) + env[i-1][1]

def within_dev(val, check, dev):
    """
    TODO
    """
    return val <= check + dev and val >= check - dev
        

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
            out.append(list(tp))
            if not (len(out[-1]) == 2 and is_num_or_none(out[-1][0]) \
                    and is_num_or_none(out[-1][1])):                
                return False, None
        return True, out
    return False, None

def is_valid_cost_range(check, ats_snd, merge_time, before_merge = True):    
    """ TODO
    snd#_frq_av_range:
        NOTE: only averages non-zero frequency values, unless all are 0.0 in that range
        If None, will use .frq_av for each partial
        If float/int will interpret that as seconds from merge_start to before the time before merge_start to use for snd1, to time after merge_start for snd2 for all partials (must be > 0.0)
        If 2 float/int iterable will use that as the time range in seconds to average over for all partials
        For other iterables:
            if fewer than the number of partials, remaining partials will use .frq_av
            if more than the number of partials, will ignore the excess
            each iterable specifies it's corresponding partial, using None, float/int or a 2 float/int iterable as stated above
    """    
    if check is None:
        return True, copy(ats_snd.frq_av)
    elif isinstance(check, (float, int)):
        if check <= 0.0:
            print("WARNING: negative range value, assuming None")
            return True, copy(ats_snd.frq_av)
        start = merge_time
        end = merge_time
        if before_merge:
            start = merge_time - check            
        else:
            end = merge_time + check
        return get_averages_in_time_range(ats_snd, start, end)

    elif not is_iterable(check):
        return False, None

    else:
        out = zeros(ats_snd.partials, "float64")
        collect = [ck for ck in check]
        if len(collect) == 0:
            print("WARNING: empty iterable, assuming None")
            return True, copy(ats_snd.frq_av)
        if len(collect) == 2:
            if isinstance(collect[0], (float, int)) and isinstance(collect[1], (float, int)):                
                # use this as the time range to average over for all partials
                return get_averages_in_time_range(ats_snd, collect[0], collect[1])

        for ind, ck in enumerate(collect[:ats_snd.partials]):            
            if ck is None:
                out[ind] = ats_snd.frq_av[ind]
            elif isinstance(ck, (float, int)):
                if ck <= 0.0:
                    print("WARNING: single negative range value, assuming None")
                    out[ind] = ats_snd.frq_av[ind]
                start = merge_time
                end = merge_time
                if before_merge:
                    start = merge_time - ck
                else:
                    end = merge_time + ck
                out[ind] = get_average_in_time_range(ats_snd, ind, start, end)

            elif not is_iterable(ck):
                return False, None
            else:
                tp = tuple(ck)
                tp_len = len(tp)
                if tp_len == 0:
                    print("WARNING: single empty tuple, assuming None")
                    out[ind] = ats_snd.frq_av[ind]
                elif tp_len != 2 or not isinstance(tp[0], (float, int)) or not isinstance(tp[1], (float, int)):
                    return False, None
                else:
                    out[ind] = get_average_in_time_range(ats_snd, ind, tp[0], tp[1])

        # do remaining partials
        for ind in range(len(collect), ats_snd.partials):
            out[ind] = ats_snd.frq_av[ind]
        return True, out

def get_average_in_time_range(ats_snd, partial, start, end):
    """TODO
    """
    start = max(0.0, start)
    if start > ats_snd.dur:
        print("WARNING: partial cost range specified started past ats sound duration")
        return 0.0
    
    end = max(min(ats_snd.dur, end), 0.0)
    if start >= end:
        print("WARNING: partial cost range specified had duration of 0.0 or was negative after constraining to ats sound duration")
        return 0.0

    time_selection = (ats_snd.time >= start) & (ats_snd.time <= end)
    frq = 0.0
    time_sum = 0.0

    if any(time_selection):
        true_indices = where(time_selection)
        start_ind = true_indices[0][0]
        end_ind = true_indices[0][-1]
        for ind in range(start_ind + 1, end_ind + 1):
            t_dur = ats_snd.time[ind] - ats_snd.time[ind - 1]
            if ats_snd.frq[partial][ind] > 0.0 and ats_snd.frq[partial][ind - 1] > 0.0:
                time_sum += t_dur
                frq += (ats_snd.frq[partial][ind] + ats_snd.frq[partial][ind - 1]) * 0.5 * t_dur
            elif ats_snd.frq[partial][ind] > 0.0:
                time_sum += t_dur
                frq += ats_snd.frq[partial][ind] * t_dur
            elif ats_snd.frq[partial][ind - 1] > 0.0:
                time_sum += t_dur
                frq += ats_snd.frq[partial][ind - 1] * t_dur
        # handle head
        if start_ind > 0:
            t_dur = ats_snd.time[start_ind] - start             
            if ats_snd.frq[partial][start_ind] > 0.0 and ats_snd.frq[partial][start_ind - 1] > 0.0:
                time_sum += t_dur
                lo_frq = ats_snd.frq[partial][start_ind] + ((ats_snd.frq[partial][start_ind - 1] - ats_snd.frq[partial][start_ind]) * (t_dur / (ats_snd.time[start_ind] - ats_snd.time[start_ind - 1])))
                frq += (lo_frq + ats_snd.frq[partial][start_ind]) * 0.5 * t_dur
            elif ats_snd.frq[partial][start_ind] > 0.0:
                time_sum += t_dur
                frq += ats_snd.frq[partial][start_ind] * t_dur
        # handle tail
        if end_ind < ats_snd.time.size - 1:
            t_dur = end - ats_snd.time[end_ind]                  
            if ats_snd.frq[partial][end_ind] > 0.0 and ats_snd.frq[partial][end_ind + 1] > 0.0:
                time_sum += t_dur
                hi_frq = ats_snd.frq[partial][end_ind] + ((ats_snd.frq[partial][end_ind + 1] - ats_snd.frq[partial][end_ind]) * (t_dur / (ats_snd.time[end_ind + 1] - ats_snd.time[end_ind])))
                frq += (hi_frq + ats_snd.frq[partial][end_ind]) * 0.5 * t_dur
            elif ats_snd.frq[partial][end_ind] > 0.0:
                time_sum += t_dur
                frq += ats_snd.frq[partial][end_ind] * t_dur

        if time_sum > 0.0:
            return frq / time_sum
        else:
            return 0.0
    else:
        # range lies completely between frames
        hi_ind = where(ats_snd.time >= end)[0][0]
        lo_ind = hi_ind - 1
        t1 = start - ats_snd.time[lo_ind]
        t2 = end - ats_snd.time[hi_ind]
        t_dur = ats_snd.time[hi_ind] - ats_snd.time[lo_ind]
        t1_scalar = t1 / t_dur
        t2_scalar = t2 / t_dur
        if ats_snd.frq[partial][lo_ind] > 0.0 and ats_snd.frq[partial][hi_ind] > 0.0:
            lo_frq = ats_snd.frq[partial][lo_ind] + (t1_scalar * (ats_snd.frq[partial][hi_ind] - ats_snd.frq[partial][lo_ind]))                    
            hi_frq = ats_snd.frq[partial][lo_ind] + (t2_scalar * (ats_snd.frq[partial][hi_ind] - ats_snd.frq[partial][lo_ind]))                    
            frq = (hi_frq + lo_frq) * 0.5
        elif ats_snd.frq[partial][lo_ind] > 0.0:
            frq = ats_snd.frq[partial][lo_ind]
        elif ats_snd.frq[partial][hi_ind] > 0.0:                    
            frq = ats_snd.frq[partial][hi_ind]
        return frq


def get_averages_in_time_range(ats_snd, start, end):
    """
    TODO
    """
    # constrain start/end to ats_snd 
    start = max(0.0, start)
    if start > ats_snd.dur:
        print("WARNING: cost range specified started past ats sound duration")
        return True, zeros(ats_snd.partials,"float64")
    
    end = max(min(ats_snd.dur, end), 0.0)
    if start >= end:
        print("WARNING: cost range specified had duration of 0.0 or was negative after constraining to ats sound duration")
        return True, zeros(ats_snd.partials,"float64")
    
    time_selection = (ats_snd.time >= start) & (ats_snd.time <= end)
    frq = zeros(ats_snd.partials, "float64")
    time_sum = zeros(ats_snd.partials, "float64")

    if any(time_selection):
        true_indices = where(time_selection)
        start_ind = true_indices[0][0]
        end_ind = true_indices[0][-1]
        
        for ind in range(start_ind + 1, end_ind + 1):
            t_dur = ats_snd.time[ind] - ats_snd.time[ind - 1]                                
            for p in range(ats_snd.partials):
                if ats_snd.frq[p][ind] > 0.0 and ats_snd.frq[p][ind - 1] > 0.0:
                    time_sum[p] += t_dur
                    frq[p] += (ats_snd.frq[p][ind] + ats_snd.frq[p][ind - 1]) * 0.5 * t_dur
                elif ats_snd.frq[p][ind] > 0.0:
                    time_sum[p] += t_dur
                    frq[p] += ats_snd.frq[p][ind] * t_dur
                elif ats_snd.frq[p][ind - 1] > 0.0:
                    time_sum[p] += t_dur
                    frq[p] += ats_snd.frq[p][ind - 1] * t_dur
        # handle head
        if start_ind > 0:
            t_dur = ats_snd.time[start_ind] - start 
            for p in range(ats_snd.partials):                       
                if ats_snd.frq[p][start_ind] > 0.0 and ats_snd.frq[p][start_ind - 1] > 0.0:
                    time_sum[p] += t_dur
                    lo_frq = ats_snd.frq[p][start_ind] + ((ats_snd.frq[p][start_ind - 1] - ats_snd.frq[p][start_ind]) * (t_dur / (ats_snd.time[start_ind] - ats_snd.time[start_ind - 1])))
                    frq[p] += (lo_frq + ats_snd.frq[p][start_ind]) * 0.5 * t_dur
                elif ats_snd.frq[p][start_ind] > 0.0:
                    time_sum[p] += t_dur
                    frq[p] += ats_snd.frq[p][start_ind] * t_dur
        # handle tail
        if end_ind < ats_snd.time.size - 1:
            t_dur = end - ats_snd.time[end_ind]
            for p in range(ats_snd.partials):                    
                if ats_snd.frq[p][end_ind] > 0.0 and ats_snd.frq[p][end_ind + 1] > 0.0:
                    time_sum[p] += t_dur
                    hi_frq = ats_snd.frq[p][end_ind] + ((ats_snd.frq[p][end_ind + 1] - ats_snd.frq[p][end_ind]) * (t_dur / (ats_snd.time[end_ind + 1] - ats_snd.time[end_ind])))
                    frq[p] += (hi_frq + ats_snd.frq[p][end_ind]) * 0.5 * t_dur
                elif ats_snd.frq[p][end_ind] > 0.0:
                    time_sum[p] += t_dur
                    frq[p] += ats_snd.frq[p][end_ind] * t_dur

        for p in range(ats_snd.partials):
            if time_sum[p] > 0.0:
                frq[p] /= time_sum[p]
            else:
                frq[p] = 0.0

        return True, frq
        
    else:
        # range lies completely between frames
        hi_ind = where(ats_snd.time >= end)[0][0]
        lo_ind = hi_ind - 1
        t1 = start - ats_snd.time[lo_ind]
        t2 = end - ats_snd.time[hi_ind]
        t_dur = ats_snd.time[hi_ind] - ats_snd.time[lo_ind]
        t1_scalar = t1 / t_dur
        t2_scalar = t2 / t_dur
        for p in range(ats_snd.partials):
            if ats_snd.frq[p][lo_ind] > 0.0 and ats_snd.frq[p][hi_ind] > 0.0:
                lo_frq = ats_snd.frq[p][lo_ind] + (t1_scalar * (ats_snd.frq[p][hi_ind] - ats_snd.frq[p][lo_ind]))                    
                hi_frq = ats_snd.frq[p][lo_ind] + (t2_scalar * (ats_snd.frq[p][hi_ind] - ats_snd.frq[p][lo_ind]))                    
                frq[p] = (hi_frq + lo_frq) * 0.5
            elif ats_snd.frq[p][lo_ind] > 0.0:
                frq[p] = ats_snd.frq[p][lo_ind]
            elif ats_snd.frq[p][hi_ind] > 0.0:                    
                frq[p] = ats_snd.frq[p][hi_ind]
        return True, frq


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
        out_env = [[(0.0, 0.0), (default_end_time, 1.0)] for p in range(partials)]
        return True, [], out_env
    elif isinstance(check, (float, int)):
        check = min(max(check, 0.0), 1.0)
        out_env = [[(0.0, check), (default_end_time, check)] for p in range(partials)]
        return True, [], out_env
    elif not is_iterable(check):
        return False, [], None
    else:
        # we have either a global envelope or a list of per-match items
        collect = []
        all_iterable = True
        for ck in check:
            if is_iterable(ck):
                item = [it for it in ck]
                collect.append(item)
            else:
                collect.append(ck)
                all_iterable = False
        # check for global envelope (i.e. collect will be all 2-item iterables of nums)
        if len(collect) > 0 and all_iterable and len(collect[0]) == 2 \
                and isinstance(collect[0][0], (int, float)) and isinstance(collect[0][1], (int, float)):
                
                cur_time = collect[0][0]
                for ck in collect[1:]:
                    if len(ck) != 2 or not isinstance(ck[0], (float, int)) \
                        or not isinstance(ck[1], (float, int)) or ck[0] <= cur_time:
                        return False, [], None
                # all checks passed, now we normalize the envelope and export to out frames
                min_time = collect[0][0] 
                max_time = collect[-1][0]
                time_norm = default_end_time / (max_time - min_time)
                collect[0][0] = 0.0
                collect[0][1] = min(max(collect[0][1], 0.0), 1.0)
                for ind in range(1, len(collect)):
                    # constrain bias
                    collect[ind][1] = min(max(collect[ind][1], 0.0), 1.0)
                    # normalize time
                    collect[ind][0] = time_norm * (collect[ind][0] - min_time)             
                    # export to out_frames
                    out_frames.append(collect[ind][0] + out_frame_time_offset)
                out_env = [collect.copy() for p in range(partials) ]
                return True, out_frames, out_env

        # otherwise we possibly have a list of per-match items
        for ck in collect:
            if ck is None:
                out_env.append([[(0.0, 0.0, (default_end_time, 1.0))]])
            elif isinstance(ck, (float, int)):
                outval = min(max(ck, 0.0), 1.0)
                out_env.append([[(0.0, outval), (default_end_time, outval)]])
            elif not is_iterable(ck):
                return False, [], None
            else:
                env = []
                cur_time = -inf
                for tp in ck:
                    # we expect tuples
                    if not is_iterable(tp):
                        return False, [], None
                    outval = list(tp)
                    # we expect 2 numbers: time, bias & time must monotonically increase
                    if len(outval) != 2 or not isinstance(outval[0], (float, int)) \
                                or not isinstance(outval[1], (float, int)) \
                                    or outval[0] <= cur_time:
                        return False, [], None
                    cur_time = outval[0]
                    outval[1] = min(max(outval[1], 0.0), 1.0)
                    env.append(list(outval))
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
                    for ind in range(1, env_len):
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

def insert_into_list_with_deviation(lst, val, dev, start_at = 0):   
    """
    TODO NOTE: may mutate input list
    """     
    ind = start_at
    new_len = len(lst)        
    while (ind < new_len):
        if lst[ind] > val:
            break
        ind += 1
    if ind == 0:
        if lst[ind] <= val + dev:
            lst[ind] = val
        else:
            lst = [val] + lst
    elif ind == new_len:
        if lst[-1] >= val - dev:
            lst[-1] = val
        else:
            lst = lst + [val]
    else:
        before = lst[ind - 1]
        after = lst[ind]
        if before >= val - dev and after <= val + dev:
            lst[ind - 1] = val
            lst = lst[:ind] + lst[ind+1:]
            ind -= 1
        elif before >= val - dev:
            lst[ind - 1] = val
            ind -= 1
        elif after <= val + dev:
            lst[ind] = val
        else:
            lst = lst[:ind] + [val] + lst[ind:]
    return ind, lst

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


if __name__ == "__main__":
    from numpy import array
    from pyatsyn.analysis.tracker import tracker
    mock1 = AtsSoundVFR(20, 8, 2.0)
    mock1.frq_av = array([400, 660, 800, 1195, 1201, 1501, 12000, 14000])

    mock2 = AtsSoundVFR(20, 5, 2.0)
    mock2.frq_av = array([400, 660, 800, 1200, 1800])

    # for match_mode in ATS_VALID_MERGE_MATCH_MODES:
    #     print(match_mode, merge(  mock1,
    #             mock2,
    #             merge_start = 1.0,
    #             merge_dur = None,
    #             ats_snd1_start = 0.0,
    #             ats_snd2_start = 0.0,
    #             ats_snd2_dur = None,           
    #             match_mode = match_mode,
    #             force_matches = [(2,4),(2,7)],
    #             snd1_frq_av_range = None,
    #             snd2_frq_av_range = None,
    #             time_deviation = None, # if None will us 1/ATS_DEFAULT_SAMPLING_RATE to account for floating point error, ignored by start/end of merge
    #             frequency_deviation = 0.1,
    #             frequency_bias_curve = None,
    #             amplitude_bias_curve = None,
    #             noise_bias_curve = None,
    #             return_match_list_only = False,
    #             ))

    # mock2 = tracker("/Users/jgl/Code/pyatsyn/sample_sounds/goat2s.wav", residual_file="/Users/jgl/Desktop/temp/merge_res1_temp.wav", verbose=True)
    # mock1 = tracker("/Users/jgl/Code/pyatsyn/sample_sounds/trumpet3s.wav", residual_file="/Users/jgl/Desktop/temp/merge_res2_temp.wav", verbose=True)

    from pyatsyn.ats_io import ats_load, ats_save
    # ats_save(mock1, "/Users/jgl/Desktop/temp/merge_mock1.ats")
    # ats_save(mock2, "/Users/jgl/Desktop/temp/merge_mock2.ats")

    mock1 = ats_load("/Users/jgl/Desktop/temp/merge_mock1.ats")
    mock2 = ats_load("/Users/jgl/Desktop/temp/merge_mock2.ats")


    merge_out = merge(  mock1,
            mock2,
            merge_start = 0.0,
            merge_dur = 2.0,
            ats_snd1_start = 0.0,
            ats_snd2_start = 0.0,
            ats_snd2_dur = None,           
            match_mode = "stable",
            force_matches = None,
            drop_unmatched = False,
            drop_unmatched_during_merge = True,
            snd1_frq_av_range = None,
            snd2_frq_av_range = None,
            time_deviation = None, # if None will us 1/ATS_DEFAULT_SAMPLING_RATE to account for floating point error, ignored by start/end of merge
            frequency_deviation = 0.2,
            frequency_bias_curve = 0, #[(0,0), (0.5, 0),(1,1)],
            amplitude_bias_curve = 1, # [(0,0), (0.5, 0), (0.8, 0), (1,1)],            
            noise_bias_curve = 1, #[(0,0), (0.5, 0),(1,1)],
            return_match_list_only = False,
            verbose = True,
            )
    #print(merge_out)

    from pyatsyn.synthesis.synth import synth
    
    synth(mock1, export_file = "/Users/jgl/Desktop/temp/merge_mock1.wav", compute_phase=True, noise_pct=0.5)
    synth(merge_out[1], export_file = "/Users/jgl/Desktop/temp/merge_out_test.wav", compute_phase=False, noise_pct=0.3, normalize_noise=True)


# TODO should merge_dur end at the end of sound 2 also???? BUG