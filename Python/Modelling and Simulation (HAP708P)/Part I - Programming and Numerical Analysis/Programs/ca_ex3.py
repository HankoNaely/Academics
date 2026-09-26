import numpy as np

N = 1000
states = np.round(np.random.random(N), 0)

def sliding_window(array, wsize):

    windows = np.tile(array, (wsize, 1))
    windows = windows.T

    windows[:,0] = np.roll(array, 1)
    windows[:,2] = np.roll(array, -1)

    return windows

sliding_window([1, 2, 3, 4, 5, 6], 3)