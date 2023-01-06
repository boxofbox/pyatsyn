from numpy import zeros, pi

from atsa_utils import amp_to_db, db_to_amp, TWO_PI
from atsa_struct import ats_peak

def peak_detection (fftfreqs, fftmags, fftphases, sample_rate, 
                    lowest_bin=None, highest_bin=None, lowest_magnitude=None, norm=1.0):
    
    peaks = []

    N = highest_bin
    if N is None:
        N = fftfreqs.size
    
    first_bin = lowest_bin
    if first_bin is None or first_bin < 1:
        first_bin = 1
    
    frqs = fftfreqs[first_bin-1:N]
    mags = fftmags[first_bin-1:N]
    phs = fftphases[first_bin-1:N]
    fq_scale = frqs[1] - frqs[0]

    for k in range(1,N-first_bin-1):

        left = mags[k-1]
        center = mags[k]
        right = mags[k+1]

        if center > lowest_magnitude and center > right and center > left:
            pk = ats_peak()
            offset, pk.amp = parabolic_interp(left, center, right)
            pk.frq = frqs[k] + (fq_scale * offset)
            if (offset > 0):
                pk.pha = phase_interp(phs[k-1], phs[k], offset)
            else:
                pk.pha = phase_interp(phs[k], phs[k+1], offset)
            peaks.append(pk)

    return peaks


def parabolic_interp(alpha, beta, gamma):
    """
    parabolic-interp <alpha> <beta> <gamma>
    does parabolic interpolation of 3 points
    returns the x offset and height
    of the interpolated peak    
    """
    dB_alpha = amp_to_db(alpha)
    dB_beta = amp_to_db(beta)
    dB_gamma = amp_to_db(gamma)
    dB_alpha_minus_gamma = dB_alpha - dB_gamma
    offset = 0.5 * dB_alpha_minus_gamma / (dB_alpha + dB_gamma + (-2 * dB_beta))
    height = db_to_amp(dB_beta - (0.25 * dB_alpha_minus_gamma * offset))
    return offset, height

def phase_interp(left, right, offset):
    """
    angular interpolation
    """  
    if left - right > 1.5 * pi:
        return (left + (offset * (right - left + TWO_PI)))
    elif right - left > 1.5 * pi:
        return (left + (offset * (right - left - TWO_PI)))
    else:
        return (left + (offset * (right - left)))