from struct import pack, unpack, calcsize
from numpy import zeros, arange

from pyats.ats_structure import AtsSound

"""
-ATS header consists of (all double floats):

ATS_MAGIC_NUMBER
sampling-rate (samples/sec)
frame-size (samples)
window-size (samples)
partials (number)
frames (number)
ampmax (max. amplitude)
frqmax (max. frequecny)
dur (duration)
type (number, see below)

-ATS frames can be of four different types:

1) without phase or noise:
==========================
time (frame starting time)
amp (par#0 amplitude)
frq (par#0 frequency)
...
amp (par#n amplitude)
frq (par#n frequency)


2) with phase but not noise:
============================
time (frame starting time)
amp (par#0 amplitude)
frq (par#0 frequency)
pha (par#0 phase)
...
amp (par#n amplitude)
frq (par#n frequency)
pha (par#n phase)


3) with noise but not phase:
============================
time (frame starting time)
amp (par#0 amplitude)
frq (par#0 frequency)
...
amp (par#n amplitude)
frq (par#n frequency)

energy (band#0 energy)
...
energy (band#n energy)

4) with phase and noise:
========================
time (frame starting time)
amp (par#0 amplitude)
frq (par#0 frequency)
pha (par#0 phase)
...
amp (par#n amplitude)
frq (par#n frequency)
pha (par#n phase)

noise (band#0 energy)
...
noise (band#n energy)
"""

ATS_MAGIC_NUMBER = 123.0
DOUBLE_SIZE = calcsize('d')
DOUBLE_BIG_ENDIAN = '>d'
DOUBLE_LIL_ENDIAN = '<d'

def ats_save(sound, file, save_phase=True, save_noise=True):

    has_pha = save_phase and len(sound.pha) > 0
    has_noi = save_noise and (len(sound.energy) > 0 or len(sound.band_energy) > 0)

    if has_noi and len(sound.bands) != 25:
        print("WARNING: outputing noise energies with other than the expected 25 critical bands")

    ats_type = 1
    if has_pha and has_noi:
        ats_type = 4
    elif not has_pha and has_noi:
        ats_type = 3
    elif has_pha and not has_noi:
        ats_type = 2

    with open(file, 'wb') as fil:
        # write header
        fil.write(pack('d',ATS_MAGIC_NUMBER))
        fil.write(pack('d',sound.sampling_rate))
        fil.write(pack('d',sound.frame_size))
        fil.write(pack('d',sound.window_size))
        fil.write(pack('d',sound.partials))
        fil.write(pack('d',sound.frames))
        fil.write(pack('d',sound.amp_max))
        fil.write(pack('d',sound.frq_max))
        fil.write(pack('d',sound.dur))
        fil.write(pack('d',ats_type))

        for frame_n in range(sound.frames):
            fil.write(pack('d',sound.time[frame_n]))
            for partial in range(sound.partials):
                fil.write(pack('d',sound.amp[partial][frame_n]))
                fil.write(pack('d',sound.frq[partial][frame_n]))
                if has_pha:
                    fil.write(pack('d',sound.pha[partial][frame_n]))
            
            if has_noi:
                for band in range(len(sound.bands)):
                    fil.write(pack('d',sound.band_energy[band][frame_n]))                    


def write_array_of_numbers_to_binary_doubles(file, arr):
    for n in arr:
        file.write(pack('d',n))


def ats_load(   name, 
                file, 
                optimize=False, 
                min_gap_size = None,
                min_segment_length = None,                     
                amp_threshold = None, 
                highest_frequency = None,
                lowest_frequency = None):

    with open(file, 'rb') as fil:
        
        # check ATS_MAGIC_NUMBER and set endian order
        check_magic_number_raw = fil.read(DOUBLE_SIZE)
        ordered_double = None        
        if unpack(DOUBLE_BIG_ENDIAN,check_magic_number_raw)[0] == ATS_MAGIC_NUMBER:
            ordered_double = DOUBLE_BIG_ENDIAN
        elif unpack(DOUBLE_LIL_ENDIAN,check_magic_number_raw)[0] == ATS_MAGIC_NUMBER:
            ordered_double = DOUBLE_LIL_ENDIAN
        else:
            raise Exception("File is not a compatible ATS format (ATS magic number was not accurate)")
        
        sampling_rate = int(unpack(ordered_double, fil.read(DOUBLE_SIZE))[0])
        frame_size = int(unpack(ordered_double, fil.read(DOUBLE_SIZE))[0])
        window_size = int(unpack(ordered_double, fil.read(DOUBLE_SIZE))[0])
        partials = int(unpack(ordered_double, fil.read(DOUBLE_SIZE))[0])
        frames = int(unpack(ordered_double, fil.read(DOUBLE_SIZE))[0])
        amp_max = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
        frq_max = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
        dur = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
        ats_type = int(unpack(ordered_double, fil.read(DOUBLE_SIZE))[0])

        has_noise = False
        if ats_type == 3 or ats_type == 4:
            has_noise = True
        has_phase = False
        if ats_type == 2 or ats_type == 4:
            has_phase = True

        ats_snd = AtsSound(name, sampling_rate, frame_size, window_size, 
                                partials, frames, dur, has_phase=has_phase)

        ats_snd.amp_max = amp_max
        ats_snd.frq_max = frq_max

        ats_snd.frq_av = zeros(partials, "float64")
        ats_snd.amp_av = zeros(partials, "float64")
        ats_snd.time = zeros(frames, "float64")
        ats_snd.frq = zeros([partials,frames], "float64")
        ats_snd.amp = zeros([partials,frames], "float64")
        if has_phase:
            ats_snd.pha = zeros([partials,frames], "float64")
        if has_noise:
            # NOTE: hard-coded for expected number of critical bands
            ats_snd.bands = arange(25)
            ats_snd.band_energy = zeros([25, frames], "float64")

        for frame_n in range(ats_snd.frames):
            ats_snd.time[frame_n] = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
            for partial in range(ats_snd.partials):
                ats_snd.amp[partial][frame_n] = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
                ats_snd.frq[partial][frame_n] = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
                if has_phase:
                    ats_snd.pha[partial][frame_n] = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]
            if has_noise:
                for band in range(len(ats_snd.bands)):
                    ats_snd.band_energy[band][frame_n] = unpack(ordered_double, fil.read(DOUBLE_SIZE))[0]

        if optimize:
            ats_snd.optimize(min_gap_size, min_segment_length, amp_threshold, highest_frequency, lowest_frequency)
        
        return ats_snd

