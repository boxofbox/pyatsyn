from numpy import zeros, multiply, roll, absolute, angle
from numpy.fft import fft, fftfreq
import soundfile as sf

from pyats.ats_structure import AtsSound

from pyats.atsa.utils import db_to_amp, next_power_of_2, compute_frames, optimize_tracks
from pyats.atsa.windows import make_fft_window, window_norm
from pyats.atsa.peak_detect import peak_detection
from pyats.atsa.critical_bands import evaluate_smr
from pyats.atsa.peak_tracking import update_track_averages, peak_tracking
from pyats.atsa.residual import compute_residual, residual_analysis

def tracker (   in_file, 
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
                min_gap_length = 3,
                min_segment_length = 3,
                last_peak_contribution = 0.0,
                SMR_continuity = 0.0,
                SMR_threshold = None,
                amp_threshold = None, # in dB
                residual_file = None,
                optimize = True,
                force_M = None, # None, or a forced window length in samples
                force_window = None, # None, or a numpy.ndarray of floats
                window_alpha = 0.5,
                window_beta = 1.0,
                verbose = True,                
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
        window = make_fft_window(window_type, M, beta=window_beta, alpha=window_alpha)
    
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
    win_samps = zeros(frames, "int64")
    # storage for lists of peaks
    analysis_frames = [None for _ in range(frames)]
    # set file pointer half a window from the first sample
    fil_ptr = st - M_over_2

    # minimum SMR
    min_smr = SMR_threshold
    if min_smr is None:
        min_smr = 0.0
    
    # guarantee that we have a valid minimum gap length
    if min_gap_length < 1:
        min_gap_length = 1

    fft_mags = None
    tracks = []

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

        data = zeros(N, "float64")
        data[front_pad:M-back_pad] = multiply(
                                            window[front_pad:M-back_pad], 
                                            in_sound[fil_ptr+front_pad:fil_ptr+M-back_pad]
                                            )

        # shift window by half of M so that phases in `data` are relatively accurate to midpoints
        data = roll(data, -M_over_2)

        # update file pointer
        fil_ptr += hop

        # get the DFT
        fd = fft(data)
        
        ##################
        # PEAK DETECTION #
        ##################

        fftfreqs = fftfreq(fd.size, 1 / sample_rate)
                
        if front_pad or back_pad:
            # apply correction for frames that hang off the edge of the input file, thus downscaling the actual amplitude            
            # multiply by additional 2.0 to account for symmetric negative frequencies
            fft_mags = absolute(fd) * 2.0 * window_norm(window[front_pad:M-back_pad])
        else:
            # multiply by additional 2.0 to account for symmetric negative frequencies
            fft_mags = absolute(fd) * 2.0 * norm 

        fftphases = angle(fd)

        peaks = peak_detection(fftfreqs, fft_mags, fftphases, lowest_bin, highest_bin, lowest_magnitude)         

        if peaks:
            # masking curve evaluation using a critical band based model
            evaluate_smr(peaks)

        #################
        # PEAK TRACKING #
        #################

        if len(tracks) > 0:
            update_track_averages(tracks, track_length, frame_n, analysis_frames, last_peak_contribution)
            peak_tracking(tracks, peaks, frame_n, analysis_frames, sample_rate, hop, frequency_deviation, SMR_continuity, min_gap_length)
        else:
            # otherwise instantiate tracks with the current frame's peaks, if any
            for pk_ind, pk in enumerate(peaks):
                pk.track = pk_ind
            tracks = [pk.clone() for pk in peaks]

        # store peaks for this current frame
        analysis_frames[frame_n] = peaks    

    ########################
    # INITIALIZE ATS SOUND #
    ########################

    if optimize:
        tracks = optimize_tracks(tracks, analysis_frames, min_segment_length, amp_threshold, highest_frequency, lowest_frequency)

    ats_snd = AtsSound(out_snd, sample_rate, hop, M, len(tracks), frames, analysis_duration, has_phase = True)

    if optimize:
        ats_snd.optimized = True
        amp_max = 0.0
        frq_max = 0.0
        for tk in tracks:
            ats_snd.frq_av[tk.track] = tk.frq
            ats_snd.amp_av[tk.track] = tk.amp
            amp_max = max(amp_max, tk.amp_max)
            frq_max = max(frq_max, tk.frq_max)
        ats_snd.amp_max = amp_max
        ats_snd.frq_max = frq_max

    # fill up with data
    for frame_n in range(frames):
        frame_time = (win_samps[frame_n] - st) / sample_rate
        ats_snd.time[frame_n] = frame_time
        for pk in analysis_frames[frame_n]:
            ats_snd.frq[pk.track][frame_n] = pk.frq
            ats_snd.amp[pk.track][frame_n] = pk.amp
            ats_snd.pha[pk.track][frame_n] = pk.pha     

    #####################
    # RESIDUAL ANALYSIS #
    #####################

    if residual_file:
        residual = compute_residual(residual_file, ats_snd, in_sound, st, nd)
        residual_analysis(residual, ats_snd, equalize=True, verbose=verbose)       

    return ats_snd


if __name__ == '__main__':
    from pyats.ats_io import ats_save, ats_load
    filename = 'trumpetc3'
    ats_save(   tracker('../sample_sounds/'+filename+'.wav',
                        filename+'.ats', 
                        verbose=False, 
                        residual_file='/Users/jgl/Desktop/'+filename+'_residual.wav',
                        ), 
                '/Users/jgl/Desktop/'+filename+'.ats', 
                save_phase=True, 
                save_noise=True
                )
    ats_load(filename, '/Users/jgl/Desktop/'+filename+'.ats', optimize=True,            
                min_gap_size = 1,
                min_segment_length = 8,                     
                amp_threshold = -20, 
                highest_frequency = 10000,
                lowest_frequency = 400)