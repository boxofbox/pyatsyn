from numpy import zeros, matmul, arange, cos, linspace, cumsum, sin, pi, real, sqrt
from numpy.random import uniform
from numpy.fft import fft, ifft
import soundfile as sf
from math import tau

from pyats.atsa.critical_bands import ATS_CRITICAL_BAND_EDGES
from pyats.atsa.utils import compute_frames


def synth(ats_snd, out_size, normalize=False, compute_phase=True, 
            export_file=None, sine_pct = 1.0, noise_pct = 0.0, noise_bands = None):

    synthesized = zeros(out_size,"float64")
    
    sample_rate = ats_snd.sampling_rate
    frame_size = ats_snd.frame_size
    frames = ats_snd.frames

    frame_size_range = frame_size
    
    if sine_pct > 0.0:
        n_partials = ats_snd.partials    
        freq_to_radians_per_sample = tau / sample_rate
        
        has_pha = compute_phase and len(ats_snd.pha) > 0
        """
        for cubic polynomial interpolation of phase
        credit: McAulay & Quatieri (1986)
        """
        alpha_beta_coeffs = zeros([2,2], "float64")
        alpha_beta_coeffs[0][0] = 3 / (frame_size**2)
        alpha_beta_coeffs[0][1] = -1 / frame_size
        alpha_beta_coeffs[1][0] = -2 / (frame_size**3)
        alpha_beta_coeffs[1][1] = 1 / (frame_size**2)
        alpha_beta_terms = zeros([2,1],"float64")

        half_T = frame_size / 2

        samps = arange(frame_size, dtype='int64')
        samps_squared = samps ** 2
        samps_cubed = samps ** 3

        prior_partial_phases = None
        if not has_pha:
            prior_partial_phases = zeros(n_partials,"float64")

        fil_ptr = 0
        for frame_n in range(frames):
            
            # constrain number of samples we write at tail end of sound
            if fil_ptr + frame_size > out_size:
                frame_size_range = fil_ptr + frame_size - out_size
            
            for partial in range(n_partials):
                if ats_snd.frq[partial][frame_n] == 0.0 and ats_snd.frq[partial][frame_n + 1] == 0.0:
                    continue

                # get amp step
                amp_0 = ats_snd.amp[partial][frame_n]
                amp_t = ats_snd.amp[partial][frame_n + 1]
                amp_step = (amp_t - amp_0) / frame_size

                # compute frequency/phase interpolation preliminaries
                w_0 = ats_snd.frq[partial][frame_n] * freq_to_radians_per_sample
                w_t = ats_snd.frq[partial][frame_n + 1] * freq_to_radians_per_sample
                
                if w_0 == 0.0:
                    w_0 = w_t
                elif w_t == 0.0:
                    w_t = w_0

                if has_pha:
                    pha_0 = ats_snd.pha[partial][frame_n]
                    pha_t = ats_snd.pha[partial][frame_n + 1]

                    """
                    cubic polynomial interpolation of phase
                    credit: McAulay & Quatieri (1986)
                    """
                    M = round((((pha_0 + (w_0 * frame_size) - pha_t) + (half_T * (w_t - w_0))) / tau))
                    alpha_beta_terms[0] = pha_t - pha_0 - (w_0 * frame_size) + (tau * M)
                    alpha_beta_terms[1] = w_t - w_0
                    alpha, beta = matmul(alpha_beta_coeffs, alpha_beta_terms)
                    synthesized[fil_ptr:fil_ptr + frame_size_range] += ((samps[:frame_size_range] * amp_step) + amp_0) * \
                                                                            cos(pha_0 + (w_0 * samps[:frame_size_range]) + 
                                                                                (alpha * samps_squared[:frame_size_range]) + 
                                                                                (beta * samps_cubed[:frame_size_range]))
                
                else:
                    # phaseless version
                    pha_0 = prior_partial_phases[partial]
                    w = cumsum(linspace(w_0, w_t, frame_size))
                    synthesized[fil_ptr:fil_ptr + frame_size_range] += ((samps[:frame_size_range] * amp_step) + amp_0) * \
                                                                            cos(w[:frame_size_range] + pha_0)
                    prior_partial_phases[partial] = pha_0 + w[-1]

            fil_ptr += frame_size

            if fil_ptr >= out_size:
                break

            synthesized *= sine_pct
        
    has_noi = noise_pct > 0.0 and len(ats_snd.band_energy) > 0

    if has_noi:
        # using white noise -> band-limited noise fft resynthesis method
        noise = zeros(out_size,"float64")
        
        window = sin(arange(sample_rate) * pi / sample_rate)**2 # using Hann window
        overlap = 0.5

        noise_hop = int(overlap * sample_rate)
        noise_M_over_2 = sample_rate // 2
        noise_frames = compute_frames(out_size, noise_hop)
        
        white_noise = uniform(-1,1, int(noise_frames * sample_rate / overlap) + 1)
        banded_noise = zeros([len(ats_snd.bands), out_size])

        # indices for refolding a symmetric fft after clearing freq bins
        bin_indices = zeros(sample_rate, "int64")
        for i in range(noise_M_over_2):
            bin_indices[i] = i
            bin_indices[-(i + 1)] = i

        

        # build band-limited noise
        if noise_bands is None:
            noise_bands = ATS_CRITICAL_BAND_EDGES
        for band in ats_snd.bands:
            lo = int(noise_bands[band])
            hi = int(noise_bands[band+1])
            
            in_ptr = 0
            out_ptr = -noise_M_over_2
            for frame_n in range(noise_frames):
                time_bins = white_noise[in_ptr:in_ptr+sample_rate] * window
                freq_bins = fft(time_bins)
                freq_bins[:lo] = 0.0
                freq_bins[hi+1:] = 0.0
                rev_fft = real(ifft(freq_bins[bin_indices]))

                front_pad = 0
                back_pad = 0
                if out_ptr < 0:
                    front_pad = -out_ptr
                if out_ptr + sample_rate >= out_size:
                    back_pad = out_ptr + sample_rate - out_size

                if not front_pad and not back_pad:    
                    banded_noise[band][out_ptr:out_ptr+sample_rate] += rev_fft
                else:
                    banded_noise[band][out_ptr+front_pad:out_ptr+sample_rate-back_pad] += rev_fft[front_pad:sample_rate-back_pad] 

                in_ptr += noise_hop
                out_ptr += noise_hop

                if out_ptr >= out_size:
                    break 

        # envelope bands TODO
        fil_ptr = 0
        for frame_n in range(frames):

            # constrain number of samples we write at tail end of sound
            if fil_ptr + frame_size > out_size:
                frame_size_range = fil_ptr + frame_size - out_size

            for band in ats_snd.bands:
                if ats_snd.band_energy[band][frame_n] == 0.0 and ats_snd.band_energy[band][frame_n + 1] == 0.0:
                    continue

                # get amp step
                amp_0 = ats_snd.band_energy[band][frame_n]
                amp_t = ats_snd.band_energy[band][frame_n + 1]
                amp_step = (amp_t - amp_0) / frame_size

                noise[fil_ptr:fil_ptr + frame_size_range] += (amp_0 + (arange(frame_size_range) * amp_step)) * \
                                                                banded_noise[band][fil_ptr:fil_ptr + frame_size_range]

            fil_ptr += frame_size

            if fil_ptr >= out_size:
                break

        synthesized += noise_pct * noise
    
    if normalize:
        gain = max(abs(synthesized))
        if gain != 1.0 and gain > 0:
            synthesized /= gain

    # export synthesized version to audio file
    if export_file is not None:
        sf.write(export_file, synthesized, ats_snd.sampling_rate)

    return synthesized  
