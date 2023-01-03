from numpy import inf, ceil, log2, pi


###################
# UTILITY CONSTANTS
###################

TWO_PI = 2 * pi


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


def next_power_of_2(num):
    '''
    return the closest power of 2 integer more than or equal to <num>
    '''
    return int(2**ceil(log2(num)))

def compute_frames(total_samps, hop, start, end):
    '''
    computes the number of frames in the specified analysis
    we want to have an extra frame at the end to prevent chopping the ending
    '''
    tmp = int(total_samps / hop)
    tmp2 = (tmp * hop) - hop + start
    if (tmp2 > end):
        return tmp
    else:
        return tmp + 1
