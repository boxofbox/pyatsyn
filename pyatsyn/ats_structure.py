# -*- coding: utf-8 -*-

# This source code is licensed under the BSD-style license found in the
# LICENSE.rst file in the root directory of this source tree. 

# pyatsyn Copyright (c) <2023>, <Johnathan G Lyon>
# All rights reserved.

# Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
# <Oscar Pablo Di Liscia, Pete Moss, and Juan Pampin>


"""Data Abstraction for ATS

"""

from numpy import zeros, inf, copy, sum, any, arange, ceil

from pyatsyn.analysis.utils import ATS_MIN_SEGMENT_LENGTH
from pyatsyn.ats_utils import phase_interp_cubic, ATS_DEFAULT_SAMPLING_RATE

class AtsPeak:
    """Data abstraction for storing single peak, single timepoint data for peak tracking

    Used primarily as a data-store during the peak tracking phase of analysis.

    Attributes
    ----------
    amp : float
        the amplitude of the peak
    frq : float
        the frequency (in Hz) of the peak
    pha : float
        the phase (in radians) of the peak
    smr : float
        the signal-to-mask ratio (in dB SPL) of the peak
    track : int
        the corresponding tracked partial the peak is assigned to
    db_spl : float
        peak amplitude in dB SPL (used during SMR evaluation)
    bark_frq : float
        frequency in bark scale (used during SMR evaluation)
    slope_r : float
        right slope of masking curve (used during SMR evaluation)
    asleep_for : int
        sleep counter (in frames) (used during peak tracking)
    duration : float
        active counter (in frames) (used during peak tracking)
    frq_max : float
        maximum frequency (used in track data during optimization)
    amp_max : float        
        maximum amplitude (used in track data during optimization)
    frq_min : float        
        minimum frequency (used in track data during optimization)
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
        """Function to return a copy of an :obj:`~pyatsyn.ats_structure.AtsSound`

        Returns
        -------
        :obj:`~pyatsyn.ats_structure.AtsPeak`
            a copy of the calling :obj:`~pyatsyn.ats_structure.AtsPeak` object
        """
        return AtsPeak(self.amp,self.frq,self.pha,self.smr,self.track,self.db_spl,
                        self.barkfrq,self.slope_r,self.asleep_for, self.duration)

    def __repr__(self):
        return f"PK: f_{self.frq} at mag_{self.amp} + {self.pha}"


class AtsSoundVFR():
    """Data abstraction for ATS sounds with variable frame rates

    Parameters
    ----------
    partials : int
        number of partials/tracks stored
    frames : int
        number of frames of analysis
    dur : float
        duration (in s) of the sound
    has_phase : bool, optional
        whether to initial phase information data structure (default: True)

    Attributes
    ----------
    partials : int
        number of partials/tracks stored
    frames : int
        number of frames of analysis
    dur : float
        duration (in s) of the sound
    amp_max : float
        maximum amplitude of the sound
    frq_max : float
        maximum frequency (in Hz) of the sound
    frq_av : ndarray[float]
        1D array of average frequency (in Hz) for each partial
    amp_av : ndarray[float]
        1D array of average amplitude for each partial
    time : float
        1D array of the time (in s) corresponding to each frame
    frq : ndarray[float]
        2D array storing frequency (in Hz) for each partial at each frame
    amp : ndarray[float]
        2D array storing amplitude for each partial at each frame
    pha : ndarray[float]
        2D array storing phase (in radians) for each partial at each frame. None if no phase information is stored.
    energy : ndarray[float]
        2D array for storing noise band energy into each partials at each frame. NOTE: Currently only implemented for legacy purposes. Empty list if no noise information is stored.
    band_energy : ndarray[float]
        2D array of noise band energies for each band at each frame. Empty list if no noise information is stored.
    bands : ndarray[int]
        1D array of unique indices to label each noise band. Empty list if no noise information is stored.
    """ 
    def  __init__ (self, frames, partials, dur, has_phase=True):        
        
        self.partials = partials
        self.dur = dur
        self.frames = frames

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


    def info(self, partials_info=False):
        """Function to print information about this object to the stdout
        
        Parameters
        ----------
        partials_info : bool, optional
            whether to include frq and amp averages about each partial in the output (default: False)    
        """
        print(f"n partials:", self.partials)
        print(f"n frames:", self.frames)
        print(f"maximum amplitude:", self.amp_max)
        print(f"maximum frequency (Hz):", self.frq_max)
        print(f"duration (s):", self.dur)

        has_noise = len(self.bands) > 0 and (len(self.band_energy) > 0 or len(self.energy) > 0)         
        has_phase = self.pha is not None

        ats_type = 1
        if has_phase and has_noise:
                ats_type = 4
        elif not has_phase and has_noise:
            ats_type = 3
        elif has_phase and not has_noise:
            ats_type = 2
        print(f"ATS frame type: ", ats_type)

        if partials_info:
            print(f"\nPartial Information:")
            for partial in range(self.partials):
                print(f"\tpartial #{partial}:\t\tfrq_av {self.frq_av[partial]:.2f}\t\tamp_av {self.amp_av[partial]:.5f}")
        

    def clone(self):
        """Function to return a deep copy of an :obj:`~pyatsyn.ats_structure.AtsSoundVBR`

        Returns
        -------
        :obj:`~pyatsyn.ats_structure.AtsSoundVBR`
            a deep copy of the calling :obj:`~pyatsyn.ats_structure.AtsSoundVBR` object
        """
        has_pha = self.pha is not None        
        
        new_ats_snd = AtsSoundVFR(self.partials, self.frames, self.dur, has_phase=has_pha)
        
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
                    min_gap_time = None,
                    min_segment_time = None,                     
                    amp_threshold = None, # in amplitude
                    highest_frequency = None,
                    lowest_frequency = None):
        """Function to run optimization routines on the frames of partial data stored in the object.

        The optimizations performed are:
            * fill gaps of min_gap_time or shorter
            * trim short partials
            * calculate and store maximum and average frq and amp
            * prune partials below amplitude threshold
            * prune partials outside frequency constraints
            * re-order partials according to average frq  

        Parameters
        ----------
        min_gap_time : float, optional
            partial gaps longer than this (in s) will not be interpolated and filled in.
            If None, this sub-optimization will be skipped.  (default: None)
        min_segment_time : float, optional
            minimal size (in s) of a valid partial segment, otherwise it is pruned. 
            If None, this sub-optimization will be skipped. (default: None)
        amp_threshold : float, optional
            amplitude threshold used to prune partials. 
            If None, this sub-optimization will be skipped. (default: None)
        highest_frequency : float
            upper frequency threshold, tracks with maxima above this will be pruned. 
            If None, this sub-optimization will be skipped. (default: None)
        lowest_frequency : float
            lower frequency threshold, tracks with minima below this will be pruned. 
            If None, this sub-optimization will be skipped. (default: None)
        """
        has_pha = self.pha is not None        
        if min_gap_time is not None:
            # fill gaps of min_gap_size or shorter
            for partial in range(self.partials):
                asleep_for = 0
                asleep_time = 0.0
                for frame_n in range(self.frames):
                    if self.amp[partial][frame_n] == 0.0:
                        asleep_for += 1
                        if asleep_for > 1:
                            asleep_time += self.time[frame_n] - self.time[frame_n - 1]
                    else:
                        if asleep_for > 0 and asleep_time <= min_gap_time:
                            fell_asleep_at = frame_n - asleep_for
                            if fell_asleep_at != 0: # skip if waking up for the first time
                                # interpolate the gap
                                interp_range = asleep_for + 1
                                earlier_ind = fell_asleep_at - 1                                                  
                                asleep_time += self.time[fell_asleep_at + 1] - self.time[fell_asleep_at]
                                asleep_time += self.time[frame_n] - self.time[frame_n - 1]
                                t = asleep_time
                                frq_step = self.frq[partial][earlier_ind] - self.frq[partial][frame_n]
                                amp_step = self.amp[partial][earlier_ind] - self.amp[partial][frame_n]
                                for i in range(1, interp_range): # NOTE: we'll walk backward from frame_n
                                    t -= self.time[frame_n - i + 1] - self.time[frame_n - i]
                                    mult = t / asleep_time
                                    self.frq[partial][frame_n - i] = (frq_step * mult) + self.amp[partial][frame_n]
                                    self.amp[partial][frame_n - i] = (amp_step * mult) + self.amp[partial][frame_n]
                                if has_pha:
                                    for i in range(1, interp_range): # NOTE: we'll walk backward from frame_n
                                        frq_0 = self.frq[partial][earlier_ind]
                                        frq_t = self.frq[partial][frame_n]
                                        pha_0 = self.pha[partial][earlier_ind]
                                        pha_t = self.pha[partial][frame_n]
                                        i_samps_from_0 = (self.time[frame_n - i] - self.time[earlier_ind]) * self.sampling_rate
                                        samps_from_0_to_t = (self.time[frame_n] - self.time[earlier_ind]) * self.sampling_rate 
                                        self.pha[partial][frame_n - i] = phase_interp_cubic(frq_0, frq_t, pha_0, pha_t, i_samps_from_0, samps_from_0_to_t, self.sampling_rate)

                        asleep_for = 0
                        asleep_time = 0.0

        keep_partials = set(arange(self.partials))
        if min_segment_time is not None:                    
            for partial in range(self.partials):
                n_segments = 0
                segment_frames = 0
                segment_time = 0.0
                for frame_n in range(self.frames):
                    if self.amp[partial][frame_n] > 0.0:
                        segment_frames += 1
                        if frame_n > 0:
                            segment_time += (self.time[frame_n] - self.time[frame_n - 1]) * 0.5
                        if frame_n < self.frames - 1:
                            segment_time += (self.time[frame_n + 1] - self.time[frame_n]) * 0.5
                    elif segment_frames > 0:
                        # we've reached the end of a segment
                        if segment_time >= min_segment_time:
                            n_segments += 1                            
                        else:
                            # remove the segment
                            for ind in range(frame_n - segment_frames, frame_n):
                                self.frq[partial][ind] = 0.0
                                self.amp[partial][ind] = 0.0
                                if has_pha:
                                    self.pha[partial][ind] = 0.0
                        # reset the segment counter
                        segment_time = 0.0
                        segment_frames = 0               
                # handle last segment        
                if segment_frames > 0:
                    if segment_time >= min_segment_time:
                        n_segments += 1
                    else:
                        # remove the segment
                        for ind in range(self.frames - segment_frames, self.frames):
                            self.frq[partial][ind] = 0.0
                            self.amp[partial][ind] = 0.0
                            if has_pha:
                                self.pha[partial][ind] = 0.0
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
            self.energy = []
            return            
        else:            
            self.frq_av = self.frq_av[keep_partials]
            self.amp_av = self.amp_av[keep_partials]
            self.frq = self.frq[keep_partials,:]
            self.amp = self.amp[keep_partials,:]
            if has_pha:
                self.pha = self.pha[keep_partials,:]
            if len(self.energy) > 0:
                self.energy = self.energy[keep_partials,:]

        # reset amp_max & average
        amp_max = 0.0
        frq_max = 0.0
        for partial in range(self.partials):
            frq_selection = self.frq[partial,:] > 0.0
            amp_selection = self.amp[partial,:] > 0.0
            
            any_f = any(frq_selection)
            any_a = any(amp_selection)
            if any_f or any_a:
                # built durations for each frame for averaging
                frame_times = zeros(self.frames, "float64")
                half_frame_time_diffs = (self.time[1:] - self.time[:self.frames-1]) * 0.5
                frame_times[:self.frames-1] = half_frame_time_diffs
                frame_times[1:] += half_frame_time_diffs
                if any_f:
                    self.frq_av[partial] = sum(self.frq[partial][frq_selection] * (self.time[frq_selection] / sum(self.time[frq_selection])))                    
                else:
                    self.frq_av[partial] = 0.0
                if any_a:
                    self.amp_av[partial] = sum(self.amp[partial][amp_selection] * (self.time[amp_selection] / sum(self.time[amp_selection])))                    
                else:
                    self.amp_av[partial] = 0.0
            frq_max = max(frq_max, max(self.frq[partial,:]))
            amp_max = max(amp_max, max(self.amp[partial,:]))            
        self.amp_max = amp_max
        self.frq_max = frq_max

        # re-sort tracks by frq_av
        partial_av_tuples = [(i, self.frq_av[i]) for i in range(self.partials)]
        sorted_tuples = sorted(partial_av_tuples, key=lambda tp: tp[1])
        sorted_index = [tp[0] for tp in sorted_tuples]

        self.frq_av = self.frq_av[sorted_index]
        self.amp_av = self.amp_av[sorted_index]
        self.frq = self.frq[sorted_index,:]
        self.amp = self.amp[sorted_index,:]
        if has_pha:
            self.pha = self.pha[sorted_index,:]


class AtsSound(AtsSoundVFR):
    """Main data abstraction for ATS

    subclass of :obj:`~pyatsyn.ats_structure.AtsSoundVFR` with a constant frame size

    Parameters
    ----------
    sampling_rate : float
        sampling rate (samples/sec)
    frame_size : int
        interframe distance (in samples)
    window_size : int
        size (in samples) of the FFT window used to analyze the sound
    partials : int
        number of partials/tracks stored
    frames : int
        number of frames of analysis
    dur : float
        duration (in s) of the sound
    has_phase : bool, optional
        whether to initial phase information data structure (default: True)

    Attributes
    ----------
    sampling_rate : float
        sampling rate (samples/sec)
    frame_size : int
        interframe distance (in samples)
    window_size : int
        size (in samples) of the FFT window used to analyze the sound
    partials : int
        number of partials/tracks stored
    frames : int
        number of frames of analysis
    dur : float
        duration (in s) of the sound
    amp_max : float
        maximum amplitude of the sound
    frq_max : float
        maximum frequency (in Hz) of the sound
    frq_av : ndarray[float]
        1D array of average frequency (in Hz) for each partial
    amp_av : ndarray[float]
        1D array of average amplitude for each partial
    time : float
        1D array of the time (in s) corresponding to each frame
    frq : ndarray[float]
        2D array storing frequency (in Hz) for each partial at each frame
    amp : ndarray[float]
        2D array storing amplitude for each partial at each frame
    pha : ndarray[float]
        2D array storing phase (in radians) for each partial at each frame. None if no phase information is stored.
    energy : ndarray[float]
        2D array for storing noise band energy into each partials at each frame. NOTE: Currently only implemented for legacy purposes. Empty list if no noise information is stored.
    band_energy : ndarray[float]
        2D array of noise band energies for each band at each frame. Empty list if no noise information is stored.
    bands : ndarray[int]
        1D array of unique indices to label each noise band. Empty list if no noise information is stored.
    """    
    def __init__ (self, sampling_rate, frame_size, window_size, 
                  partials, frames, dur, has_phase=True):
        
        super().__init__(frames, partials, dur, has_phase)

        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.window_size = window_size


    def info(self, partials_info=False):
        """Function to print information about this object to the stdout
        
        Parameters
        ----------
        partials_info : bool, optional
            whether to include frq and amp averages about each partial in the output (default: False)    
        """
        print(f"sampling rate (samples/s):", self.sampling_rate)
        print(f"frame size:", self.frame_size)
        print(f"window size:", self.window_size)
        super().info(partials_info)


    def clone(self):
        """Function to return a deep copy of an :obj:`~pyatsyn.ats_structure.AtsSound`

        Returns
        -------
        :obj:`~pyatsyn.ats_structure.AtsSound`
            a deep copy of the calling :obj:`~pyatsyn.ats_structure.AtsSound` object
        """
        has_pha = self.pha is not None
        
        new_ats_snd = AtsSound(self.sampling_rate, self.frame_size, self.window_size, 
                  self.partials, self.frames, self.dur, has_phase=has_pha)
                  
        new_ats_snd.amp_max = self.amp_max
        new_ats_snd.frq_max = self.frq_max
        
        new_ats_snd.time = copy(self.time)
        new_ats_snd.frq = copy(self.frq)
        new_ats_snd.amp = copy(self.amp)
        new_ats_snd.frq_av = copy(self.frq_av)
        new_ats_snd.amp_av = copy(self.amp_av)
        
        if has_pha:
            new_ats_snd.pha = copy(self.pha)

        if len(self.bands) > 0:
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
        """Function to run optimization routines on the frames of partial data stored in the object.

        The optimizations performed are:
            * fill gaps of min_gap_size or shorter
            * trim short partials
            * calculate and store maximum and average frq and amp
            * prune partials below amplitude threshold
            * prune partials outside frequency constraints
            * re-order partials according to average frq  

        Parameters
        ----------
        min_gap_size : int, optional
            partial gaps longer than this (in frames) will not be interpolated and filled in.
            If None, this sub-optimization will be skipped.  (default: None)
        min_segment_length : int, optional
            minimal size (in frames) of a valid partial segment, otherwise it is pruned. 
            If None, this sub-optimization will be skipped. (default: None)
        amp_threshold : float, optional
            amplitude threshold used to prune partials. 
            If None, this sub-optimization will be skipped. (default: None)
        highest_frequency : float
            upper frequency threshold, tracks with maxima above this will be pruned. 
            If None, this sub-optimization will be skipped. (default: None)
        lowest_frequency : float
            lower frequency threshold, tracks with minima below this will be pruned. 
            If None, this sub-optimization will be skipped. (default: None)
        """
        if min_segment_length is not None:
            if min_segment_length < 1:
                min_segment_length = ATS_MIN_SEGMENT_LENGTH

        frame_to_t_conversion = self.frame_size / self.sampling_rate

        min_gap_time = min_gap_size
        if min_gap_time is not None:
            min_gap_time = (min_gap_size * frame_to_t_conversion) + (1 / self.sampling_rate) # padding for floating point inaccuracy

        min_segment_time = min_segment_length
        if min_segment_time is not None:
            min_segment_time = (min_segment_length * frame_to_t_conversion) - (1 / self.sampling_rate) # padding for floating point inaccuracy 

        super().optimize(   min_gap_time,
                            min_segment_time,
                            amp_threshold, 
                            highest_frequency, 
                            lowest_frequency
                            )


def to_cfr(ats_snd_vfr, frame_size, sampling_rate=None, window_size=None):
    """Function to convert :obj:`~pyatsyn.ats_structure.AtsSoundVFR` to constant frame rate :obj:`~pyatsyn.ats_structure.AtsSound`

    Can also be used to resample a :obj:`~pyatsyn.ats_structure.AtsSound` at another frame rate.

    Parameters
    ----------
    ats_snd_vfr : :obj:`~pyatsyn.ats_structure.AtsSoundVFR` or :obj:`~pyatsyn.ats_structure.AtsSound`
        the ats sound object to resample
    frame_size : int
        interframe distance (in samples) to be used to dictate the new constant frame rate
    sampling_rate : float, optional
        sampling rate (samples/sec). If None, will be assigned to :obj:`~pyatsyn.ats_utils.ATS_DEFAULT_SAMPLING_RATE` (Default: None)
    window_size : int, optional
        size (in samples) of the FFT window used to analyze the sound. If None, will be assigned to -1 (Default: None)

    Returns
    -------
    :obj:`~pyatsyn.ats_structure.AtsSound`
        an interpolated or resamples ats sound object with constant frame rate
    """    
    has_pha = ats_snd_vfr.pha is not None
    if sampling_rate is None:
        sampling_rate = ATS_DEFAULT_SAMPLING_RATE
    if window_size is None:
        window_size = -1

    frames = int(ceil((ats_snd_vfr.time[-1] * sampling_rate) / frame_size)) + 1

    ats_cfr = AtsSound(sampling_rate, frame_size, window_size, ats_snd_vfr.partials, frames, ats_snd_vfr.dur, has_phase=has_pha)

    if len(ats_snd_vfr.bands) > 0:
        ats_cfr.bands = copy(ats_snd_vfr.bands)
        if len(ats_snd_vfr.band_energy) > 0:
            ats_cfr.band_energy = zeros([ats_cfr.bands.size, frames], "float64")
    if len(ats_snd_vfr.energy) > 0:
        ats_cfr.energy = zeros([ats_cfr.partials, frames], "float64")

    vfr_frame_n = 0
    for frame_n in range(frames):
        ats_cfr.time[frame_n] = frame_n * frame_size / sampling_rate

        while (ats_snd_vfr.time[vfr_frame_n] < ats_cfr.time[frame_n] and vfr_frame_n < ats_snd_vfr.frames - 1):
            vfr_frame_n += 1
        
        # if the old frame is exactly at the current time or at the beginning/end of the input frames, just copy it over
        if vfr_frame_n == 0 or vfr_frame_n == ats_snd_vfr.frames - 1 or ats_snd_vfr.time[vfr_frame_n] == ats_cfr.time[frame_n]:
            ats_cfr.frq[:,frame_n] = copy(ats_snd_vfr.frq[:,vfr_frame_n])
            ats_cfr.amp[:,frame_n] = copy(ats_snd_vfr.amp[:,vfr_frame_n])
            if has_pha:
                ats_cfr.pha[:,frame_n] = copy(ats_snd_vfr.pha[:,vfr_frame_n])            
            if len(ats_snd_vfr.band_energy) > 0:
                ats_cfr.band_energy[:, frame_n] = copy(ats_snd_vfr.band_energy[:,vfr_frame_n])
            if len(ats_snd_vfr.energy) > 0:
                ats_cfr.energy[:, frame_n] = copy(ats_snd_vfr.energy[:,vfr_frame_n])
                
        # otherwise interpolate a new frame
        else:
            tt = (ats_snd_vfr.time[vfr_frame_n] - ats_snd_vfr.time[vfr_frame_n - 1])
            ti = (ats_cfr.time[frame_n] - ats_snd_vfr.time[vfr_frame_n - 1])
            t_interp =  ti / tt
            i_samps_from_0 = ti * sampling_rate
            samps_from_0_to_t = tt * sampling_rate
            for partial in range(ats_cfr.partials):
                frq_0 = ats_snd_vfr.frq[partial][vfr_frame_n - 1]
                frq_t = ats_snd_vfr.frq[partial][vfr_frame_n]
                ats_cfr.frq[partial][frame_n] = ((frq_t - frq_0) * (t_interp)) + frq_0
                amp_0 = ats_snd_vfr.amp[partial][vfr_frame_n - 1]
                amp_t = ats_snd_vfr.amp[partial][vfr_frame_n]
                ats_cfr.amp[partial][frame_n] = ((amp_t - amp_0) * (t_interp)) + amp_0
                if has_pha:
                    pha_0 = ats_snd_vfr.pha[partial][vfr_frame_n - 1]
                    pha_t = ats_snd_vfr.pha[partial][vfr_frame_n]                  
                    ats_cfr.pha[partial][frame_n] = phase_interp_cubic(frq_0, frq_t, pha_0, pha_t, i_samps_from_0, samps_from_0_to_t, sampling_rate)
                if len(ats_cfr.energy) > 0:
                    eng_0 = ats_snd_vfr.energy[partial][vfr_frame_n - 1]
                    eng_t = ats_snd_vfr.energy[partial][vfr_frame_n]
                    ats_cfr.energy[partial][frame_n] = ((eng_t - eng_0) * (t_interp)) + eng_0
            if len(ats_cfr.band_energy) > 0:
                for band in range(ats_cfr.bands.size):
                    noi_0 = ats_snd_vfr.band_energy[band][vfr_frame_n - 1]
                    noi_t = ats_snd_vfr.band_energy[band][vfr_frame_n]
                    ats_cfr.band_energy[band][frame_n] = ((noi_t - noi_0) * (t_interp)) + noi_0
                
    # optimize with default None parameters to update max/av values and reorder partials if necessary
    ats_cfr.optimize()

    return ats_cfr

class MatchCost():
    """Object to abstract cost for comparisons

    Attributes
    ----------
    cost : float
        the calculated cost to `index`
    index : int
        the index that indicates the track the cost was calculated against
    """ 
    def __init__(self, cost, index):
        self.cost = cost
        self.index = index
    
    def __repr__(self):
        return f"to index: {self.index} at cost: {self.cost}"
    
    def __lt__(self, other):
        return self.cost < other.cost
