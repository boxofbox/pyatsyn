from numpy import inf, ceil, log2, pi, log10


###################
# UTILITY CONSTANTS
###################

TWO_PI = 2 * pi
MAX_DB_SPL = 100.0


###################
# UTILITY FUNCTIONS
###################

def db_to_amp(db):
    '''
    convert decibels to amplitude
    '''
    if (db == -inf):
        return 0.0
    return pow(10, (db / 20.0))

def amp_to_db(amp):
    '''
    convert amplitude to decibels
    '''
    return 20 * log10(amp)

def amp_to_db_spl(amp):
    return MAX_DB_SPL + amp_to_db(amp)

def next_power_of_2(num):
    '''
    return the closest power of 2 integer more than or equal to <num>
    '''
    return int(2**ceil(log2(num)))

def compute_frames(total_samps, M_over_2, hop, start, end):
    '''
    computes the number of frames in the specified analysis
    we want to have an extra frame at the end to prevent chopping the ending
    '''
    tmp = (total_samps + M_over_2) // hop # frame 0 begins half a window before 'start'
    tmp2 = (tmp * hop) - hop + start
    if (tmp2 > end):
        return tmp
    else:
        return tmp + 1
