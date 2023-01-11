REQUIRES LIBSNDFILE

atsa.util.py
* for amp thresholding is it better to use amp_av or amp_max?

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

atsa_tracker.py
* clean-up numpy imports



IDEAS
* online ATS, optimize dropping tracks, and improving default sorted state


1.0 TODO
* finish residual calc: gain, phaseless, st/en/dur
* residual analysis
* ats_save/ats_load & re-optimize
* double check phase interpolation scheme in peak_tracking.py
* test windows
* clean-up/minimal imports
* rebuild requirements.txt
* test & deploy package mode
** how do we tell it we need libsndfile?