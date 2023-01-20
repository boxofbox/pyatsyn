from numpy import zeros, cos, sin, log, exp, sqrt, absolute, ones, pi
from math import tau


"""
All data coming form Harris' famous paper:
"On the Use Of windows For Harmonic Analysis 
 With The Discrete Fourier Transform"
Proceedings of the IEEE, Vol. 66, No. 1 (pg. 51 to 84)
January 1978
    and
Albert H. Nuttall, "Some Windows with Very Good Sidelobe Behaviour", 
IEEE Transactions of Acoustics, Speech, and Signal Processing, Vol. ASSP-29,
No. 1, February 1981, pp 84-91
"""

VALID_FFT_WINDOW_DEFINITIONS = [
    'blackman-exact',
    'blackman',
    'blackman-harris-3-1',
    'blackman-harris-3-2',
    'blackman-harris-4-1',
    'blackman-harris-4-2',
    'rectangular',
    'parzen',
    'welch',
    'kaiser',
    'gaussian',
    'poisson',
    'cauchy',
    'connes',
    'welch',
    'kaiser',
    'gaussian',
    'poisson',
    'cauchy',
    'connes',
    'exponential',
    'bartlett',
    'riemann',
    'tukey',
    'hamming',
    'hann',
    'hann-poisson',
]  


# Window coefficients (a0, a1, a2, a3)
ATS_BLACKMAN_WINDOW_COEFF = zeros([6,4], dtype='float64')
ATS_BLACKMAN_WINDOW_COEFF[0] = [0.42659, -0.49656, 0.07685, 0]      # Exact Blackman (-51 dB)
ATS_BLACKMAN_WINDOW_COEFF[1] = [0.42, -0.5, 0.08, 0]            # Blackman (rounded coeffs) (-58 dB)
ATS_BLACKMAN_WINDOW_COEFF[2] = [0.42323, -0.49755, 0.07922, 0]       # 3-term Blackman-Harris 1 (-67 dB)
ATS_BLACKMAN_WINDOW_COEFF[3] = [0.44959, -0.49364, 0.05677, 0]       # 3-term Blackman-Harris 2 (-61 dB)
ATS_BLACKMAN_WINDOW_COEFF[4] = [0.35875, -0.48829, 0.14128, -0.01168]   # 4-term Blackman-Harris 1 (-92 dB)
ATS_BLACKMAN_WINDOW_COEFF[5] = [0.40217, -0.49703, 0.09392, -0.00183]  # 4-term Blackman-Harris 2 (-71 dB)

ATS_BLACKMAN_WINDOW_COEFF_LABELS = {
    'blackman-exact': ATS_BLACKMAN_WINDOW_COEFF[0],
    'blackman': ATS_BLACKMAN_WINDOW_COEFF[1],
    'blackman-harris-3-1': ATS_BLACKMAN_WINDOW_COEFF[2],
    'blackman-harris-3-2': ATS_BLACKMAN_WINDOW_COEFF[3],
    'blackman-harris-4-1': ATS_BLACKMAN_WINDOW_COEFF[4],
    'blackman-harris-4-2': ATS_BLACKMAN_WINDOW_COEFF[5]
    }

def make_blackman_window(window_type, size):
    if window_type not in ATS_BLACKMAN_WINDOW_COEFF_LABELS.keys():
        raise Exception('Specified Blackman Window Type not Defined')

    coeffs =  ATS_BLACKMAN_WINDOW_COEFF_LABELS[window_type]

    two_pi_over_size = tau / size
    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]
    a3 = coeffs[3]
    win = zeros(size,'float64')
    for count in range(0,size):
        coeff4 = 0
        if (a3 != 0):
            coeff4 = a3 * cos(3 * two_pi_over_size * count)
        win[count] = a0 + (a1 * cos(two_pi_over_size * count)) + (a2 * cos(2 * two_pi_over_size * count)) + coeff4
    return win
 

def make_fft_window(window_type, size, beta=1.0, alpha=0.5):
    if (window_type.startswith('blackman')):
        return make_blackman_window(window_type, size)

    midn = size // 2

    if window_type == 'rectangular':
        return ones(size, dtype="float64")
    elif window_type == 'parzen' or window_type == 'welch' or window_type == 'kaiser' or \
            window_type == 'gaussian' or window_type == 'poisson' or window_type == 'cauchy' \
                or window_type == 'connes':
        window = zeros(size, 'float64')        
        midp1 = (size + 1) // 2
        win_fun = lambda x: 1.0 - abs( (x - midn) / midp1) # parzen
        if window_type == 'welch':
            win_fun = lambda x: 1.0 - pow(((x - midn) / midp1), 2)
        elif window_type == 'kaiser':
            win_fun = lambda x: bes_i0((beta * (sqrt(1.0 - pow((midn - x) / midn, 2))))) / bes_i0(beta)
        elif window_type == 'gaussian':
            win_fun = lambda x: exp( -0.5 * pow((beta * ((midn - x)/ midn)), 2))
        elif window_type == 'poisson':
            win_fun = lambda x: exp( -beta * ((midn - x) / midn))
        elif window_type == 'cauchy':
            win_fun = lambda x: 1.0 / (1.0 + pow(((beta * (midn - x)) / midn), 2))
        elif window_type == 'connes':
            win_fun = lambda x: pow((1.0 - pow( ((x - midn) / midp1), 2)), 2)
        for i in range(0, midn + 1):
            val = win_fun(i)
            window[i] = val
            window[-(i+1)] = val            
        return window
    elif window_type == 'exponential':
        window = zeros(size, 'float64')
        expsum = 1.0
        expn = (1.0 + (log(2) / midn))
        for i in range(0, midn + 1):
            val = expsum - 1.0
            expsum *= expn
            window[i] = val
            window[-(i+1)] = val 
        return window 
    elif window_type == 'bartlett':
        window = zeros(size, 'float64')
        rate = 1.0 / midn
        angle = 0.0
        for i in range(0, midn + 1):
            val = angle
            angle += rate
            window[i] = val
            window[-(i+1)] = val                  
        return window
    elif window_type == 'riemann':
        window = zeros(size, 'float64')
        sr = tau / size
        for i in range(0, midn): 
            val = sin(sr * (midn - i)) / (sr * (midn - i))
            window[i] = val
            window[-(i+1)] = val
        window[midn] = 1                
        return window        
    elif window_type == 'tukey':        
        window = zeros(size, 'float64')
        knee = alpha * size // 2
        for i in range(0, midn + 1):
            if i >= knee:
                val = 1.0
            else:
                val = 0.5 * (1.0 - cos( (pi * i) / knee))
            window[i] = val
            window[-(i+1)] = val            
        return window
    elif window_type == 'hamming' or window_type == 'hann' or window_type == 'hann-poisson':
        window = zeros(size, 'float64')
        freq = tau / size
        angle = 0.0
        win_fun = lambda x: 0.54 - (0.46 * x) # hamming
        if window_type == 'hann':
            win_fun = lambda x: 0.5 - (0.5 * x)
        elif window_type == 'hann-poisson':
            win_fun = lambda x: (0.5 - (0.5 * cx)) * exp( -beta * ((midn - i) / midn))
        for i in range(0, midn + 1):
            cx = cos(angle)
            val = win_fun(cx)
            window[i] = val
            window[-(i+1)] = val 
            angle += freq           
        return window
    else:
        raise Exception('Specified Window Type not Defined')

# Returns the norm of the window
def window_norm (window):
    norm_factor = absolute(window).sum()
    if norm_factor == 0.0:
        raise Exception('Cannot normalize window with absolute sum of 0.0')
    return 1.0 / norm_factor
    
def norm_window(window):
    """
    returns a normalized window (i.e., integrates to 1.0)
    """
    window_sum = sum(window)
    if window_sum == 0.0:
        window_sum += 1 / len(window)
    else:
        window /= window_sum
    return window


# Modified Bessel Function of the First Kind
# from "Numerical Recipes in C"
def bes_i0 (x):
    if (abs(x) < 3.75):
        y = pow( (x / 3.75), 2)
        return (1.0 + (y * (3.5156229 + (y * (3.0899414 + \
            (y * (1.2067492 + (y * (0.2659732 + (y * \
            (0.360768e-1 + (y * 0.45813e-2))))))))))))
    else:
        ax = abs(x)
        y = 3.75 / ax
        return ( (exp(ax) / sqrt(ax)) * (0.39894228 + \
            (y * (0.1328592e-1 + (y * (0.225319e-2 + (y * \
            (-0.157565e-2 + (y * (0.916281e-2 + (y * \
            (-0.2057706e-1 + (y * (0.2635537e-1 + (y * \
            (-0.1647633e-1 + (y * 0.392377e-2)))))))))))))))))
