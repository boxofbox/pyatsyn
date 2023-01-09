REQUIRES LIBSNDFILE
REMOVE UNECESSARY REQUIREMENTS

in general, phase calculations are an issue
    * we are assuming interpolatable phase, because we are assuming continuous partials
    * however, there is not guarantee the truth is smoothly linear
    * thus later samples of phase via the fft may not adhere to this model
    * if we interpolate phase we may encounter discontinuities with this
    * current assumption is that the fft derived phases are accurate
    * however, interpolated phases (e.g., during gap fills) are not guarantee to be so
    * the are built based on the assumed correctness of the previously obtained fft phase

atsa_peak_tracking.py
* the way we are finding peaks to update averages in previous frames can be faster?
* possible speed up in inital cost calculations if we sort and terminate once we're out of range?

atsa_peak_detect.py
* phase interpolation is not accurate

atsa_windows.py
* Potentially replace with scipy.signal.windows
* TEST ALL WINDOWS

atsa.py
* clean-up numpy imports

IDEAS
* online ATS, optimize dropping tracks, and improving default sorted state
