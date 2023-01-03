
from numpy import zeros, cos, log, exp, sqrt, absolute

from atsa_utils import TWO_PI

"""
All data coming form Harris' famous paper:
"On the Use Of windows For Harmonic Analysis 
 With The Discrete Fourier Transform"
Proceedings of the IEEE, Vol. 66, No. 1 (pg. 51 to 84)
January 1978
Albert H. Nuttall, "Some Windows with Very Good Sidelobe Behaviour", 
IEEE Transactions of Acoustics, Speech, and Signal Processing, Vol. ASSP-29,
No. 1, February 1981, pp 84-91
"""


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

    two_pi_over_size = TWO_PI / size
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

FFT_WINDOW_DEFINITIONS = {
        'rectangular' : [0, 'val=1.0'],
        'parzen' : [0, 'val=(1.0 - abs( float(i - midn) / midp1))'],
        'welch' : [0, 'val=(1.0 - pow((float(i - midn) / midp1), 2))'],
        'kaiser' : [0, 'val=(bes_i0((beta * (sqrt(1.0 - pow(float(midn - i) / midn, 2))))) / I0beta)'],
        'gaussian' : [0, 'val=(exp( -0.5 * pow((beta * (float(midn - i)/ midn)), 2)))'],
        'poisson' : [0, 'val=(exp( -beta * (float(midn - i) / midn)))'],
        'cauchy' : [0, 'val=(1.0 / (1.0 + pow(((beta * float(midn - i)) / midn), 2)))'],
        'connes' : [0, 'val=pow((1.0 - pow( (float(i - midn) / midp1), 2)), 2)'],    
        'exponential' : [1, 'val=(expsum - 1.0)', 'expsum=(expsum * expn)'],
        'bartlett' : [1, 'val=angle', 'angle=(angle + rate)'],
        'riemann' : [2, 'midn==i', 'val=1.0', 'val=(sin(sr * (midn - i)) / (sr * (midn - i)))'],
        'tukey' : [3, 'pos=(midn * (1.0 - beta))', 'i >= pos', 'val=1.0', 'val=(0.5 * (1.0 - cos( (pi * i) / pos)))'],
        'hamming' : [4, 'val=(0.54 - (0.46 * cx))'],
        'hann' : [4, 'val=(0.5 - (0.5 * cx))'],
        'hann-poisson' : [4,'val=((0.5 - (0.5 * cx)) * exp( -beta * (float(midn - i) / midn)))']
        }    


def make_fft_window(window_type, size, beta=1.0, mu=0.0):
    if (window_type.startswith('blackman')):
        return make_blackman_window(window_type, size)
    if window_type not in FFT_WINDOW_DEFINITIONS.keys():
        raise Exception('Specified Window Type not Defined')
    
    param_switch = FFT_WINDOW_DEFINITIONS[window_type][0]

    window = zeros(size, 'float64')
   
    midn = size // 2
    midp1 = (size + 1) // 2
    freq = TWO_PI / size
    rate = 1.0 / midn
    sr = TWO_PI / size
    angle = 0.0
    expn = (1.0 + (log(2) / midn))
    expsum = 1.0
    IObeta = 0.0
    val = 0.0
    j = size - 1

    if (param_switch==0):
        if window_type == 'kaiser':
            I0beta = bes_i0(beta)
        for i in range(0,midn+1):
            exec(FFT_WINDOW_DEFINITIONS[window_type][1])
            window[i] = val
            window[j] = val
            j = j - 1
    elif (param_switch==1):
        for i in range(0,midn+1):
            exec(FFT_WINDOW_DEFINITIONS[window_type][1])
            exec(FFT_WINDOW_DEFINITIONS[window_type][2])
            window[i] = val
            window[j] = val
            j = j - 1
    elif (param_switch==2):
        for i in range(0,midn+1):
            if (eval(FFT_WINDOW_DEFINITIONS[window_type][1])):
                exec(FFT_WINDOW_DEFINITIONS[window_type][2])
            else:
                exec(FFT_WINDOW_DEFINITIONS[window_type][3])
            window[i] = val
            window[j] = val
            j = j - 1
    elif (param_switch==3):
        exec(FFT_WINDOW_DEFINITIONS[window_type][1])
        for i in range(0,midn+1):
            if (eval(FFT_WINDOW_DEFINITIONS[window_type][2])):
                exec(FFT_WINDOW_DEFINITIONS[window_type][3])
            else:
                exec(FFT_WINDOW_DEFINITIONS[window_type][4])
            window[i] = val
            window[j] = val
            j = j - 1
    elif (param_switch==4):
        for i in range(0,midn+1):
            cx = cos(angle)
            exec(FFT_WINDOW_DEFINITIONS[window_type][1])
            window[i] = val
            window[j] = val
            j = j - 1
            angle = angle + freq

    return window


# Returns the norm of the window
def window_norm (window):
    norm_factor = absolute(window).sum()
    if norm_factor == 0.0:
        raise Exception('Cannot normalize window with absolute sum of 0.0')
    return 1.0 / norm_factor
    
    
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

