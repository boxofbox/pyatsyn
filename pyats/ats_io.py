from struct import pack

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

def ats_save(sound, file, save_phase=True, save_noise=True):

    has_pha = save_phase and len(sound.pha) > 0
    has_noi = save_noise and (sound.energy or sound.band_energy)

    type = 1
    if has_pha and has_noi:
        type = 4
    elif not has_pha and has_noi:
        type = 3
    elif has_pha and not has_noi:
        type = 2

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
        fil.write(pack('d',type))

        for frame_n in range(sound.frames):
            fil.write(pack('d',sound.time[frame_n]))
            for partial in range(sound.partials):
                fil.write(pack('d',sound.amp[partial][frame_n]))
                fil.write(pack('d',sound.frq[partial][frame_n]))
                if has_pha:
                    fil.write(pack('d',sound.pha[partial][frame_n]))
            
            # TODO
            # write noise
            if has_noi:
                pass

    

def write_array_of_numbers_to_binary_doubles(file, arr):
    for n in arr:
        file.write(pack('d',n))


