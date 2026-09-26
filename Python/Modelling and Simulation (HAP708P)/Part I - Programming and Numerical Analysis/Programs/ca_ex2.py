import numpy as np

def cell_states_to_int(array):
    
    powers = 2 ** np.arange(np.shape(array)[-1]-1, -1, -1)

    intg = array @ powers

    return intg
