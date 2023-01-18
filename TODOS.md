atsa.peak_tracking.py
* the way we are finding peaks to update averages in previous frames can be faster?
* possible speed up in inital cost calculations if we sort and terminate once we're out of range?
* in general, phase calculations are an issue
    * we are assuming interpolatable phase, because we are assuming continuous partials
    * however, there is not guarantee the truth is smoothly linear
    * thus later samples of phase via the fft may not adhere to this model
    * if we interpolate phase we may encounter discontinuities with this
    * current assumption is that the fft derived phases are accurate
    * however, interpolated phases (e.g., during gap fills) are not guarantee to be so
    * the are built based on the assumed correctness of the previously obtained fft phase

1.0 TODO
* re-write and test windows.py
* fix window_norm
* ats synth for noise
* add verbosity & remove debug flag
* rebuild requirements.txt
* documentation
* test & deploy package mode
* how do we tell setuptools we need libsndfile?

BUGBIN
* start end of residual has a blip for the sine test ?due to inaccurate phase calc at beginning & end?
* the consequence of par_energy=True is not clearly specified in legacy code, thus it is not properly implemented. Band size will no longer be 25, and the energy is not output with the ats.file
* current ats standard silently assumes 25 critical bands for noise, but this is not strictly required in the pyats environment
* i suspect the window norm is not entirely accurate, b/c for a rectangular window it applies a non x1.0 gain change
* equalization of the residual energy using Parseval's thereom leads to significantly quieter default noise synthesis in supercollider than is expected compared to the true residual

OPEN Qs
* for amp thresholding in atsa_util.py is it better to use amp_av or amp_max?
* in the residual analysis we do not re-perform smr evaluation on the frame, but in the original CL code this was done. Which is correct?

IDEAS
* numpy-ify and add numba for every major calculation loop
* online ATS, optimize dropping tracks, and improving default sorted state
* set operations when hybridizing
* ats aware xfade (maybe a logic/reaper extension?)
* scour through PV & ATS methods in csound to build CLI versions
* scour throug the old cl code for additional functionality to implement
* vst player with MIDI control (piano roll triggering?)