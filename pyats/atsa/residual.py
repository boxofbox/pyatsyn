from numpy import zeros, mean, roll, arange, absolute
from numpy.fft import fft
import soundfile as sf

from pyats.atsa.utils import TWO_PI, next_power_of_2, db_to_amp, ATS_NOISE_THRESHOLD
from pyats.atsa.critical_bands import ATS_CRITICAL_BAND_EDGES
from pyats.ats_synth import synth

def compute_residual(   residual_file, 
                                ats_snd, 
                                in_sound,
                                start_sample,
                                end_sample,
                                export_residual = True,
                                ):
    """
    Computes the difference between the ats_snd synthesis and the original sound
    """

    synthesized = synth(ats_snd, end_sample - start_sample)
    residual = in_sound[start_sample:end_sample] - synthesized

    # export residual to audio file
    if export_residual:
        sf.write(residual_file, residual, ats_snd.sampling_rate)

    return residual


def residual_analysis(  residual, 
                        ats_snd, 
                        min_fft_size = 4096,
                        equalize = False,
                        pad_factor = 2,
                        band_edges = None,
                        par_energy = None,
                        verbose = True,                                               
                        ):
    
    hop = ats_snd.frame_size
    M = ats_snd.window_size
    M_over_2 = (M - 1) // 2
    window_gain = 1 / M
    norm = 1.0 # we will use a rectangular window of area 1
    
    threshold = db_to_amp(ATS_NOISE_THRESHOLD)

    N = residual_N(M, min_fft_size, pad_factor)

    frames = ats_snd.frames
    fft_mag = ats_snd.sampling_rate / N

    # sub band limits
    if band_edges is None:
        band_edges = ATS_CRITICAL_BAND_EDGES
    band_limits = residual_get_band_limits(fft_mag, band_edges)
    n_bands = len(band_limits) - 1
    band_energy = zeros([n_bands,frames],"float64")

    if verbose:
        print(f"Beginning Residual FFT... [frames = {frames}; M = {M}; N = {N}]")
        report_every_x_percent = 10
    report_flag = 0

    fft_mags = None
    fil_ptr = -M_over_2

    for frame_n in range(frames):

        if verbose:
            done = frame_n * 100.0 / frames
            if done > report_flag:
                print(f"{done}% complete (residual analysis)")
                report_flag += report_every_x_percent
        
        # padding for window ranges that are out of the input file
        front_pad = 0
        back_pad = 0
        if fil_ptr < 0:
            front_pad = -fil_ptr
        if fil_ptr + M >= residual.size:
            back_pad = fil_ptr + M - residual.size
        
        data = zeros(N, "float64")
        # windowed data
        data[front_pad:M-back_pad] = residual[fil_ptr+front_pad:fil_ptr+M-back_pad] * window_gain

        # shift window by half of M so that phases in `data` are relatively accurate to midpoints
        data = roll(data, -M_over_2)
        
        # update file pointer
        fil_ptr += hop

        # DC Block
        data = data - mean(data)

        # FFT
        fd = fft(data)

        if front_pad or back_pad:
            # apply correction for frames that hang off the edge of the input file, thus downscaling the actual amplitude            
            # multiply by additional 2.0 to account for symmetric negative frequencies
            adjusted_norm = 1 / (M - back_pad - front_pad)
            fft_mags = absolute(fd) * 2.0 * adjusted_norm
        else:
            # multiply by additional 2.0 to account for symmetric negative frequencies
            fft_mags = absolute(fd) * 2.0 * norm 

        residual_compute_band_energy(fft_mags, band_limits, band_energy, frame_n)

        if equalize:
            # re-scale frequency band energy to the energy in the time domain
            t_domain_energy = sum(data**2) # via Parseval's Theorem         
            f_domain_energy = sum(band_energy[:,frame_n])
            eq_ratio = 1.0
            if f_domain_energy > 0.0:
                eq_ratio = t_domain_energy / f_domain_energy
            band_energy[:,frame_n] /= eq_ratio
            

    # apply noise threshold
    band_energy[band_energy < threshold] = 0.0

    # store in ats object
    ats_snd.band_energy = band_energy
    ats_snd.bands = arange(n_bands, dtype='int64')

    if par_energy:
        band_to_energy(ats_snd, band_edges)
        remove_bands(ats_snd, threshold)


def residual_N(M, min_fft_size, factor = 2):
    if M * factor > min_fft_size:
        return next_power_of_2(M * factor)
    else:
        return next_power_of_2(min_fft_size)


def residual_get_band_limits(fft_mag, band_edges):
    band_limits = zeros(len(band_edges),"int64")
    for ind, band in enumerate(band_edges):
        band_limits[ind] = band / fft_mag
    return band_limits 


def residual_compute_band_energy(fft_mags, band_limits, band_energy, frame_n):

    for band in range(len(band_limits) - 1):

        low = band_limits[band]
        if low < 0:
            low = 0
        high = band_limits[band + 1]
        if high > fft_mags.size // 2:
            high = fft_mags.size // 2

        band_energy[band][frame_n] = sum(fft_mags[low:high]**2) / fft_mags.size


def band_to_energy(ats_snd, band_edges):    
    bands = len(ats_snd.bands)
    partials = ats_snd.partials
    frames = ats_snd.frames
    par_energy = zeros([partials,frames],"float64")
    
    for frame_n in range(frames):
        pass
    # TODO


# TODO
def remove_bands(ats_snd, threshold):
    pass


        
"""


(defun get-band-partials (lo hi sound frame)
  "returns a list of partial numbers that fall 
in frequency between lo and hi
"
  (let ((par nil))
    (loop for k from 0 below (ats-sound-partials sound) do 
      (if (<= lo (aref (aref (ats-sound-frq sound) k) frame) hi)
	  (push k par)))
    (nreverse par)))


(defun band-to-energy (sound &key (use-smr NIL)(debug NIL))
"
transfers band energy to partials
"
  (let* ((bands (if (ats-sound-bands sound)(length (ats-sound-bands sound)) *ats-critical-bands*))
	 (partials (ats-sound-partials sound))
	 (frames (ats-sound-frames sound))
	 (par-energy (make-array partials :element-type 'array)))
    ;;; create storage place for partial energy
    (loop for i from 0 below partials do
      (setf (aref par-energy i) (make-double-float-array frames :initial-element 0.0)))
    ;;; now compute par-energy frame by frame
    (loop for frame from 0 below frames do
      (let ((smr (if use-smr (smr-frame sound frame) nil)))
	(loop for b from 0 below (1- bands) do
	  (let* ((lo-frq (nth b *ats-critical-band-edges*))
		 (hi-frq (nth (1+ b) *ats-critical-band-edges*))
		 (par (get-band-partials lo-frq hi-frq sound frame))
		 (band-energy (aref (aref (ats-sound-band-energy sound) b) frame)))
	    ;;; if we found partials in this band evaluate the energy
	    (if (and (> band-energy 0.0) par)
		(let* ((par-amp-sum (loop for p in par sum 
				      (if smr (aref smr p)
					(aref (aref (ats-sound-amp sound) p) frame))))
		       (n-pars (list-length par)))
		  ;;; check if we have active partials and store band-energy proportionally
		  (if (> par-amp-sum 0.0)
		      (loop for p in par do
			(setf (aref (aref par-energy p) frame) 
			      (/ (* (if smr (aref smr p)
				      (aref (aref (ats-sound-amp sound) p) frame)) band-energy)
				 par-amp-sum)))
		    ;;; inactive partials: split energy by partials
		    (loop 
		      for p in par 
		      with eng = (/ band-energy n-pars)
		      do
		      (setf (aref (aref par-energy p) frame) eng)))
		  ;;; clear energy from band
		  (setf (aref (aref (ats-sound-band-energy sound) b) frame) (double-float 0.0))
		  )
	      (if (and debug (> band-energy 0.0))
		  (format t "Frame: ~d Band: ~d Energy: ~a no partials~%" frame b band-energy)))))))
      (setf (ats-sound-energy sound) par-energy)))



(defun energy-to-band (sound band frame)
  "
transfers energy from partials to a band
"
  (let* ((lo-frq (nth band *ats-critical-band-edges*))
	 (hi-frq (nth (1+ band) *ats-critical-band-edges*))
	 (par (get-band-partials lo-frq hi-frq sound frame)))
    (loop for p in par sum
      (aref (aref (ats-sound-energy sound) p) frame))))



;;; removes bands that have average energy below threshold
(defun remove-bands (sound &optional (threshold *ats-noise-threshold*))
  (let* ((frames (ats-sound-frames sound))
	 (threshold (db-amp threshold))
	 (band-l (get-valid-bands sound threshold))
	 (new-bands (make-array (list-length band-l) :element-type 'array)))
    ;;; now we only keep the bands we want 
    (loop 
      for i in band-l
      for k from 0
      do
      (setf (aref new-bands k)
	    (aref (ats-sound-band-energy sound) i)))
    ;;; finally we store things in the sound
    (setf (ats-sound-band-energy sound) new-bands)
    (setf (ats-sound-bands sound)(coerce band-l 'array))))
"""