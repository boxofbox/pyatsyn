from numpy import zeros, cos, arange, matmul
import soundfile as sf

from atsa.utils import TWO_PI

def compute_residual(   residual_file, 
                                ats_snd, 
                                in_sound,
                                start_sample
                                ):
    """
    Computes the difference between the ats_snd synthesis and the original sound
    """

    # TODO: phase-less version

    synthesized = zeros(in_sound.size,"float64")
    fil_ptr = start_sample
    sample_rate = ats_snd.sampling_rate
    frame_size = ats_snd.frame_size
    n_partials = ats_snd.partials
    frames = ats_snd.frames

    freq_to_radians_per_sample = TWO_PI / sample_rate

    frame_size_range = frame_size
    
    alpha_beta_coeffs = zeros([2,2], "float64")
    alpha_beta_coeffs[0][0] = 3 / (frame_size**2)
    alpha_beta_coeffs[0][1] = -1 / frame_size
    alpha_beta_coeffs[1][0] = -2 / (frame_size**3)
    alpha_beta_coeffs[1][1] = 1 / (frame_size**2)
    alpha_beta_terms = zeros([2,1],"float64")

    half_T = frame_size / 2

    samps = arange(frame_size)
    samps_squared = samps ** 2
    samps_cubed = samps ** 3

    for frame_n in range(frames):
        
        # constrain number of samples we write at tail end of sound
        if fil_ptr + frame_size > in_sound.size:
            frame_size_range = in_sound.size - fil_ptr - frame_size
        
        for partial in range(n_partials):
            if ats_snd.frq[partial][frame_n] == 0.0 and ats_snd.frq[partial][frame_n + 1] == 0.0:
                continue

            # compute frequency/phase interpolation preliminaries
            w_0 = ats_snd.frq[partial][frame_n] * freq_to_radians_per_sample
            w_t = ats_snd.frq[partial][frame_n + 1] * freq_to_radians_per_sample
            
            if w_0 == 0.0:
                w_0 = w_t
            elif w_t == 0.0:
                w_t = w_0

            pha_0 = ats_snd.pha[partial][frame_n]
            pha_t = ats_snd.pha[partial][frame_n + 1]

            """
            cubic polynomial interpolation of phase
            credit: McAulay & Quatieri (1986)
            """
            M = round((((pha_0 + (w_0 * frame_size) - pha_t) + (half_T * (w_t - w_0))) / TWO_PI))
            alpha_beta_terms[0] = pha_t - pha_0 - (w_0 * frame_size) + (TWO_PI * M)
            alpha_beta_terms[1] = w_t - w_0
            alpha, beta = matmul(alpha_beta_coeffs, alpha_beta_terms)

            # get amp step
            amp_0 = ats_snd.amp[partial][frame_n]
            amp_t = ats_snd.amp[partial][frame_n + 1]
            amp_step = (amp_t - amp_0) / frame_size

            synthesized[fil_ptr:fil_ptr + frame_size_range] += ((samps[:frame_size_range] * amp_step) + amp_0) * \
                                                                    cos(pha_0 + (w_0 * samps[:frame_size_range]) + 
                                                                        (alpha * samps_squared[:frame_size_range]) + 
                                                                        (beta * samps_cubed[:frame_size_range]))            

        # TODO gain

        fil_ptr += frame_size

        if fil_ptr >= in_sound.size:
            break     

    # TODO fix for when st/en/dur are not consistent with input file

    sf.write(residual_file, in_sound - synthesized, sample_rate)

    return synthesized


def residual_analysis(residual):
    pass