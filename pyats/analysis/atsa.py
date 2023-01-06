import numpy as np
import soundfile as sf

from atsa_utils import db_to_amp, next_power_of_2, compute_frames
from atsa_windows import make_fft_window, window_norm
from atsa_peak_detect import peak_detection
from atsa_critical_bands import evaluate_smr
from atsa_peak_tracking import update_tracks, peak_tracking

# TODO: PLOTTING UTILITIES FOR DEBUG ONLY - DELETE LATER
import matplotlib.pyplot as plt

def analyze (   in_file, 
                out_snd,
                start = 0.0, # analysis start point (in seconds)
                duration = None, # max duration to analyze (in seconds) or 'None' if analyze to end
                lowest_frequency = 20, # must be > 0
                highest_frequency = 20000.0,
                frequency_deviation = 0.1,
                window_cycles = 4,
                window_type = None, # defaults to 'blackman-harris-4-1'
                hop_size = 0.25, # in fraction of window size
                fft_size = None, # None, or force an fft size
                lowest_magnitude = db_to_amp(-60),
                track_length = 3,
                min_segment_length = 3,
                last_peak_contribution = 0.0,
                SMR_continuity = 0.0,
                SMR_threshold = None,
                amp_threshold = None,
                residual = None,
                par_energy = True,
                optimize = True,
                debug = False,
                verbose = False,
                force_M = None, # None, or a forced window length in samples
                force_window = None, # None, or a numpy.ndarray of floats
                window_mu = 0.0,
                window_beta = 1.0
                ):
    
    # read input audio file
    in_sound, sample_rate = sf.read(in_file)

    if in_sound.ndim > 1:
        raise Exception("Input audio file must be mono")
    
    # get first and last sample indices
    st = int(start * sample_rate)
    nd = in_sound.size
    if duration is not None:
        nd = st + int(duration * sample_rate)
    
    # calculate windowing parameters
    total_samps = nd - st
    analysis_duration = total_samps / sample_rate
    cycle_samps = int((1 / lowest_frequency) * window_cycles * sample_rate)

    M = force_M
    if M is None:
        # by default we want an odd length window centered at time 0.0
        if ( (cycle_samps % 2) == 0):
            M = cycle_samps + 1
        else:
            M = cycle_samps

    N = fft_size
    if N is None:
        # default fft size is next power of 2
        N = next_power_of_2(M)

    # instantiate window    
    window = force_window
    if window is None:
        if window_type is None:
            window_type='blackman-harris-4-1'
        window = make_fft_window(window_type, M, beta=window_beta, mu=window_mu)
    
    norm = window_norm(window)   
    hop = int(M * hop_size)
    # central point of the window
    M_over_2 = (M - 1) // 2
    frames = compute_frames(total_samps, M_over_2, hop, st, nd)    

    # magic number for fft frequencies (frequency resolution)
    fft_mag = sample_rate / N

    l_frq = lowest_frequency
    if l_frq <= 0.0:
        raise Exception('Lowest frequency must be greater than 0.0')
    
    h_frq = highest_frequency
    if h_frq > (sample_rate / 2.0):
        if verbose: print('WARNING: Capping highest frequency to Nyquist Frequency')
        h_frq = int(sample_rate / 2.0)
    if h_frq < l_frq:
        raise Exception('Highest frequency must be greater than lowest frequency')

    # lowest/highest bins to read
    lowest_bin = int(l_frq / fft_mag)
    highest_bin = int(h_frq / fft_mag)

    # used to store central points of the windows
    win_samps = np.zeros(frames, "int64")
    # storage for lists of peaks
    analysis_frames = [None for _ in range(frames)]
    # first point in fft buffer to write
    first_point = N - M_over_2
    # set file pointer half a window from the first sample
    fil_ptr = st - M_over_2

    # minimum SMR
    min_smr = SMR_threshold
    if min_smr is None:
        min_smr = 0.0

    tracks = []
    n_partials = 0

    if verbose:
        print(f"frames = {frames}")
        print(f"M = {M}; N = {N}")
        print(f"Beginning FFT -> peak tracking analysis...")
        report_every_x_percent = 10
    report_flag = 0

    for frame_n in range(frames):
        if verbose:
            done = frame_n * 100.0 / frames
            if done > report_flag:
                print(f"{done}% complete (tracking)")
                report_flag += report_every_x_percent
        
        # store in_sound sample at the middle of the window
        win_samps[frame_n] = fil_ptr + M_over_2

        ###################
        # WINDOWING + FFT #
        ###################

        # padding for window ranges that are out of the input file
        front_pad = 0
        back_pad = 0
        if fil_ptr < 0:
            front_pad = -fil_ptr
        if fil_ptr + M >= in_sound.size:
            back_pad = fil_ptr + M - in_sound.size

        data = np.zeros(N,"float64")
        data[front_pad:M-back_pad] = np.multiply(
                                            window[front_pad:M-back_pad], 
                                            in_sound[fil_ptr+front_pad:fil_ptr+M-back_pad]
                                            )

        # shift window by half of M so that phases in `data` are relatively accurate to midpoints
        data = np.roll(data, -M_over_2)

        # update file pointer
        fil_ptr += hop

        # get the DFT
        fd = np.fft.fft(data)
        
        ##################
        # PEAK DETECTION #
        ##################

        fftfreqs = np.fft.fftfreq(fd.size, 1 / sample_rate)
        fftmags = np.absolute(fd) * 2.0 * norm # multiply by 2.0 to account for symmetric negative frequencies
        fftphases = np.angle(fd)

        peaks = peak_detection(fftfreqs, fftmags, fftphases, sample_rate, lowest_bin, highest_bin, lowest_magnitude, norm)         

        if peaks:
            # masking curve evaluation using a critical band based model
            evaluate_smr(peaks)

        #################
        # PEAK TRACKING # # IN PROGRESS
        #################

        # only if was at least two frames
        if frame_n > 0:
            if tracks:
                tracks = update_tracks(tracks, track_length, frame_n, analysis_frames, last_peak_contribution)
            else:
                tracks = [pk.clone() for pk in analysis_frames[frame_n - 1]]

        if len(tracks) > 0:
            unmatched_old, unmatched_new = peak_tracking(tracks, peaks, frequency_deviation, SMR_continuity)

            # kill unmatched peaks from previous frame by adding a silent peak in this frame (death trajectory)
            for pk in unmatched_old:
                new_pk = pk.clone()
                new_pk.amp = 0.0
                new_pk.smr = 0.0
                peaks.append(new_pk)
            
            # set track and add a silent copy of new unmatched peak to previous frame (birth trajectory)
            for pk in unmatched_new:
                pk.track = n_partials
                n_partials += 1
                new_pk = pk.clone()
                new_pk.amp = 0.0
                new_pk.smr = 0.0
                analysis_frames[frame_n - 1].append(new_pk)
        else:
            # otherwise give all peaks a track number
            for pk in peaks:
                pk.track = n_partials
                n_partials += 1                

        # store peaks for this current frame
        analysis_frames[frame_n] = peaks

    ########################
    # INITIALIZE ATS SOUND # TODO
    ########################

    ############
    # OPTIMIZE # TODO
    ############

    #####################
    # RESIDUAL ANALYSIS # TODO
    #####################



if __name__ == '__main__':
    analyze('../sample_sounds/sine440.wav','cougar.ats', debug=True, verbose=True)



#       ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#       ;;; Peak Detection:
#       ;;; get peaks (amplitudes normalized by window norm)
#       ;;; list of peaks is sorted by frequency
#       (setf peaks (peak-detection fft-struct 
# 				  :lowest-bin lowest-bin 
# 				  :highest-bin highest-bin 
# 				  :lowest-magnitude lowest-magnitude 
# 				  :norm norm))
#       ;;; process peaks
#       (when peaks 
# 	;;; evaluate masking values (SMR) of peaks
# 	(evaluate-smr peaks)
# 	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# 	;;; Peak Tracking:
# 	;;; try to match peaks
# 	;;; only if we have at least 2 frames
# 	;;; and if we have active tracks 
# 	(if (and (> frame-n 0) 
# 		 (setf tracks (update-tracks tracks track-length frame-n ana-frames last-peak-contribution)))
# 	    (let ((cpy-peak nil))
#         ;;; track peaks and get leftover
# 	      (setf unmatched-peaks (peak-tracking (sort (copy-seq tracks) #'> :key #'ats-peak-smr) 
# 						   peaks frequency-deviation SMR-continuity))
#  	;;; kill unmatched peaks from previous frame
# 	      (dolist (k (first unmatched-peaks))
# 	      ;;; we copy the peak into this frame but with amp 0.0 
# 	      ;;; this represents our death trajectory
# 		(setf cpy-peak (copy-ats-peak k)
# 		      (ats-peak-amp cpy-peak) 0.0
# 		      (ats-peak-smr cpy-peak) 0.0)
# 		(push cpy-peak peaks))
#           ;;; give birth to peaks from new frame
# 	      (dolist (k (second unmatched-peaks))
# 	  ;;; set track number of unmatched peaks
# 		(setf (ats-peak-track k) n-partials)
# 		(incf n-partials)
# 	      ;;; we copy the peak into the previous frame but with amp 0.0 
# 	      ;;; this represents our born trajectory
# 		(setf cpy-peak (copy-ats-peak k)
# 		      (ats-peak-amp cpy-peak) 0.0
# 		      (ats-peak-smr cpy-peak) 0.0)
# 		(push cpy-peak (aref ana-frames (1- frame-n)))
# 		(push (copy-ats-peak k) tracks)))
# 	    ;;; give number to all peaks
# 	  (dolist (k (sort (copy-seq peaks) #'< :key #'ats-peak-frq))
# 	    (setf (ats-peak-track k) n-partials)
# 	    (incf n-partials)))
# 	(setf (aref ana-frames frame-n) peaks))
#       ;;; update file pointer
#       (setf filptr (+ (- filptr M) hop))
#       (if verbose (format t "<Frame:~d Time:~4,3F Tracks:~4,3F> " frame-n tmp n-partials)))
#     ;;; Initialize ATS sound
#     (init-sound sound 
# 		:sampling-rate file-sampling-rate
# 		:frame-size hop
# 		:window-size M
# 		:frames frames 
# 		:duration file-duration 
# 		:partials n-partials)
#     ;;; and fill it up with data
#     (loop for k from 0 below n-partials do
#       (loop for frame from 0 below frames do
# 	(let ((pe (find k (aref ana-frames frame) :key #'ats-peak-track)))
# 	  (if pe
# 	      (setf (aref (aref (ats-sound-amp sound) k) frame)(double-float (ats-peak-amp pe))
# 		    (aref (aref (ats-sound-frq sound) k) frame)(double-float (ats-peak-frq pe))
# 		    (aref (aref (ats-sound-pha sound) k) frame)(double-float (ats-peak-pha pe))))
# 	  ;;; set time anyways
# 	  (setf (aref (aref (ats-sound-time sound) k) frame)
# 		(double-float (/ (- (aref win-samps frame) st) file-sampling-rate))))))
#     ;;; finally optimize and declare new sound in ATS
#     (if optimize 
# 	(optimize-sound sound
# 			:min-frq lowest-frequency 
# 			:max-frq highest-frequency
# 			:min-length (if min-segment-length
# 					min-segment-length
# 				      *ats-min-segment-length*)
# 			:amp-threshold (if amp-threshold
# 					   amp-threshold
# 					 *ats-amp-threshold*)
# 			:verbose verbose))
#     (if verbose (format t "Partials: ~d Frames: ~d~%" (ats-sound-partials sound)(ats-sound-frames sound)))
#     ;;; register sound in the system
#     (add-sound sound)
#   ;;; now get the residual
#     (when residual
#       (compute-residual fil residual sound win-samps file-sampling-rate verbose)
#       (residual-analysis residual sound :par-energy par-energy :verbose verbose :debug debug :equalize t))
#     (close-input fil)))

# """