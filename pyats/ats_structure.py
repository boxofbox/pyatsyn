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

from numpy import zeros, inf, copy, mean, any, arange

from pyats.atsa.utils import ATS_MIN_SEGMENT_LENGTH
from pyats.atsa.peak_tracking import phase_interp

class AtsPeak:
    """TODO

    """
    def __init__ (self, amp=0.0, frq=0.0, pha=0.0, smr=0.0, track=0, db_spl=0.0, 
                  barkfrq=0.0, slope_r=0.0, asleep_for=None, duration=1):
        self.amp = amp
        self.frq = frq
        self.pha = pha
        self.smr = smr        
        self.track = track
        self.db_spl = db_spl
        self.barkfrq = barkfrq
        self.slope_r = slope_r
        self.asleep_for = asleep_for
        self.duration = duration
        self.frq_max = 0.0
        self.amp_max = 0.0    
        self.frq_min = inf

    def clone (self):
        """TODO

        """
        return AtsPeak(self.amp,self.frq,self.pha,self.smr,self.track,self.db_spl,
                        self.barkfrq,self.slope_r,self.asleep_for, self.duration)

    def __repr__(self):
        """TODO

        """
        return f"PK: f_{self.frq} at mag_{self.amp} + {self.pha}"

class AtsSound:
    """TODO
    main data abstraction
    amp, frq, and pha contain sinusoidal modeling information as arrays of
    arrays of data arranged by partial par-energy and band-energy hold
    noise modeling information (experimental format)
    """    
    def __init__ (self, name, sampling_rate, frame_size, window_size, 
                  partials, frames, dur, has_phase=True):
        """TODO

        """
        self.name = name
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.window_size = window_size
        self.partials = partials
        self.frames = frames
        self.dur = dur

        self.optimized = False
        self.amp_max = 0.0
        self.frq_max = 0.0
        self.frq_av = zeros(partials, "float64")
        self.amp_av = zeros(partials, "float64")
        self.time = zeros(frames, "float64")
        self.frq = zeros([partials, frames], "float64")
        self.amp = zeros([partials, frames], "float64")
        self.pha = None
        if has_phase:
            self.pha = zeros([partials, frames], "float64")
        # Noise Data
        self.energy = []        
        self.band_energy = []        
        self.bands = []

    
    def clone(self):
        """TODO

        """
        """
        returns a deep copy of the AtsSound
        """
        has_pha = True
        if self.pha is None:
            has_pha = False
        
        new_ats_snd = AtsSound(self.name, self.sampling_rate, self.frame_size, self.window_size, 
                  self.partials, self.frames, self.dur, has_phase=has_pha)
        
        new_ats_snd.optimized = self.optimized
        new_ats_snd.amp_max = self.amp_max
        new_ats_snd.frq_max = self.frq_max
        
        new_ats_snd.time = copy(self.time)
        new_ats_snd.frq = copy(self.frq)
        new_ats_snd.amp = copy(self.amp)
        new_ats_snd.frq_av = copy(self.frq_av)
        new_ats_snd.amp_av = copy(self.amp_av)
        
        if has_pha:
            new_ats_snd.pha = copy(self.pha)

        if self.bands:
            new_ats_snd.bands = copy(self.bands)
            new_ats_snd.band_energy = copy(self.band_energy)
            if self.energy:
                new_ats_snd.energy = copy(self.energy)            

        return new_ats_snd


    def optimize(   self, 
                    min_gap_size = None,
                    min_segment_length = None,                     
                    amp_threshold = None, # in amplitude
                    highest_frequency = None,
                    lowest_frequency = None):
        """TODO

        """

        has_pha = True
        if self.pha is None:
            has_pha = False

        if min_gap_size is not None:
            # fill gaps of min_gap_size or shorter
            for partial in range(self.partials):
                asleep_for = 0            
                for frame_n in range(self.frames):
                    if self.amp[partial][frame_n] == 0.0:
                        asleep_for += 1
                    else:
                        if asleep_for <= min_gap_size:
                            fell_asleep_at = frame_n - asleep_for
                            if fell_asleep_at != 0: # skip if waking up for the first time
                                # interpolate the gap
                                interp_range = asleep_for + 1
                                earlier_ind = fell_asleep_at - 1
                                frq_step = self.frq[partial][earlier_ind] - self.frq[partial][frame_n]
                                amp_step = self.amp[partial][earlier_ind] - self.amp[partial][frame_n]
                                for i in range(1, interp_range): # NOTE: we'll walk backward from frame_n
                                    mult = i / interp_range
                                    self.frq[partial][frame_n - i] = (frq_step * mult) + self.amp[partial][frame_n]
                                    self.amp[partial][frame_n - i] = (amp_step * mult) + self.amp[partial][frame_n]
                                if has_pha:
                                    for i in range(1, interp_range): # NOTE: we'll walk backward from frame_n
                                        freq_now = self.frq[partial][frame_n - i]
                                        freq_then = self.frq[partial][earlier_ind]
                                        pha_then = self.pha[partial][earlier_ind]
                                        t = (interp_range - i) / self.sampling_rate
                                        self.pha[partial][frame_n - i] = phase_interp(freq_then, freq_now, pha_then, t)
                        asleep_for = 0

        keep_partials = set(arange(self.partials))

        if min_segment_length is not None:            
            # trim short partials
            if min_segment_length < 1:
                min_segment_length = ATS_MIN_SEGMENT_LENGTH
            
            for partial in range(self.partials):
                n_segments = 0
                segment_dur = 0
                for frame_n in range(self.frames):
                    if self.amp[partial][frame_n] > 0.0:
                        segment_dur += 1
                    elif segment_dur > 0:
                        # we've reached the end of a segment
                        if segment_dur >= min_segment_length:
                            n_segments += 1                            
                        else:
                            # remove the segment
                            for ind in range(frame_n - segment_dur, frame_n):
                                self.frq[partial][ind] = 0.0
                                self.amp[partial][ind] = 0.0
                                if has_pha:
                                    self.pha[partial][ind] = 0.0
                        # reset the segment counter
                        segment_dur = 0                       

                if n_segments == 0:
                    keep_partials = keep_partials - {partial}
                    
        
        # process amp and/or frequency thresholds
        if amp_threshold is not None:
            for partial in range(self.partials):
                if max(self.amp[partial,:]) < amp_threshold:
                    keep_partials = keep_partials - {partial}
        if highest_frequency is not None:
            for partial in range(self.partials):
                if max(self.frq[partial,:]) > highest_frequency:
                    keep_partials = keep_partials - {partial}
        if lowest_frequency is not None:
            for partial in range(self.partials):
                selection = self.frq[partial,:] > 0.0
                if any(selection):
                    if min(self.frq[partial][selection]) < lowest_frequency:
                        keep_partials = keep_partials - {partial}
                else:
                    keep_partials = keep_partials - {partial}

        # keep only unfiltered partials 
        keep_partials = list(keep_partials)
        self.partials = len(keep_partials)
        if self.partials == 0:
            print("WARNING: optimization has removed all partials from AtsSound")
            self.frq_av = None
            self.amp_av = None
            self.frq = None
            self.amp = None
            self.pha = None
            self.amp_max = 0.0
            self.frq_max = 0.0
            return            
        else:            
            self.frq_av = self.frq_av[keep_partials]
            self.amp_av = self.amp_av[keep_partials]
            self.frq = self.frq[keep_partials,:]
            self.amp = self.amp[keep_partials,:]
            if has_pha:
                self.pha = self.pha[keep_partials,:]

        # reset amp_max & average
        amp_max = 0.0
        frq_max = 0.0
        for partial in range(self.partials):
            frq_selection = self.frq[partial,:] > 0.0
            if any(frq_selection):
                self.frq_av[partial] = mean(self.frq[partial][self.frq[partial] > 0.0])
            else:
                self.frq_av[partial] = 0.0
            amp_selection = self.amp[partial,:] > 0.0
            if any(amp_selection):
                self.amp_av[partial] = mean(self.amp[partial][self.amp[partial] > 0.0])
            else:
                self.amp_av[partial] = 0.0
            frq_max = max(frq_max, max(self.frq[partial,:]))
            amp_max = max(amp_max, max(self.amp[partial,:]))            
        self.amp_max = amp_max
        self.frq_max = frq_max

        # re-sort tracks by frq_av
        partial_av_tuples = [(i, self.frq_av[i]) for i in range(self.partials)]
        sorted_tuples = sorted(partial_av_tuples, key=lambda tp: tp[0])
        sorted_index = [tp[0] for tp in sorted_tuples]

        self.frq_av = self.frq_av[sorted_index]
        self.amp_av = self.amp_av[sorted_index]
        self.frq = self.frq[sorted_index,:]
        self.amp = self.amp[sorted_index,:]
        if has_pha:
            self.pha = self.pha[sorted_index,:]
        