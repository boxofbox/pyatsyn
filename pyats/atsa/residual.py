from numpy import zeros, cos
import soundfile as sf

from atsa.utils import TWO_PI


def get_M(ph1, frq1, ph, frq, buffer_size):
    return round((((ph1 + (frq1 * buffer_size) - ph) + (0.5 * buffer_size * (frq - frq1))) / TWO_PI))

def get_aux(ph1, ph, frq1, buffer_size, M):
    return (ph + (TWO_PI * M)) - (ph1 + (frq1 * buffer_size))

def get_alpha(aux, frq1, frq, buffer_size):
    return ((3 / (buffer_size**2)) * aux) - ((frq - frq1) / buffer_size)

def get_beta(aux, frq1, frq, buffer_size):
    return ((-2 / (buffer_size**3)) * aux) + ((frq - frq1) / (buffer_size**2))

def interp_phase(ph1, frq1, alpha, beta, i):
    """
    cubic polynomial interpolation of phase
    credit: McAulay & Quatieri (1986)
    via PARSHL paper by Smith & Serra (ca. 1993)
    """
    return ph1 + (frq1 * i) + (alpha * i * i) + (beta * i * i * i)

def compute_residual(   residual_file, 
                        ats_snd, 
                        in_sound,
                        start_sample
                        ):
    """
    Computes the difference between the ats_snd synthesis and the original sound
    """
    

    synthesized = zeros(in_sound.size,"float64")
    fil_ptr = start_sample
    sample_rate = ats_snd.sampling_rate
    frame_size = ats_snd.frame_size
    n_partials = ats_snd.partials
    frames = ats_snd.frames

    freq_to_radians_per_sample = TWO_PI / sample_rate

    for frame_n in range(frames):
        
        # TODO correct this hack
        if fil_ptr + frame_size > in_sound.size:
            frame_size = in_sound.size - fil_ptr - frame_size
            if frame_size <= 0:
                break
        
        for partial in range(n_partials):
            # compute frequency/phase interpolation preliminaries
            w_0 = ats_snd.frq[partial][frame_n] * freq_to_radians_per_sample
            w_t = ats_snd.frq[partial][frame_n + 1] * freq_to_radians_per_sample

            if w_0 == 0.0 and w_t == 0.0:
                continue
            elif w_0 == 0.0:
                w_0 = w_t
            elif w_t == 0.0:
                w_t = w_0

            pha_0 = ats_snd.pha[partial][frame_n]
            pha_t = ats_snd.pha[partial][frame_n + 1]

            M = get_M(pha_0, w_0, pha_t, w_t, frame_size)
            aux = get_aux(pha_0, pha_t, w_0, frame_size, M)
            alpha = get_alpha(aux, w_0, w_t, frame_size)
            beta = get_beta(aux, w_0, w_t, frame_size)

            # get amp step
            amp_0 = ats_snd.amp[partial][frame_n]
            amp_t = ats_snd.amp[partial][frame_n + 1]
            amp_step = (amp_t - amp_0) / frame_size

            # TODO numpy-ify
            for samp in range(frame_size):
                synthesized[fil_ptr + samp] += ((samp * amp_step) + amp_0) * cos(interp_phase(pha_0, w_0, alpha, beta, samp))

        # TODO gain

        fil_ptr += frame_size     

    sf.write(residual_file, in_sound - synthesized, sample_rate)

    return synthesized


def residual_analysis(residual):
    pass