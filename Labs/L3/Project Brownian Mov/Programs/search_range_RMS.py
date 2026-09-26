import numpy as np
from scipy.spatial import cKDTree

'''
D            # µm²/s  — rough diffusion coefficient
dt           # s      — frame interval (1 / FPS)
pixel_size   # µm/px
p_miss       # acceptable fraction of displacements to miss (1%)
'''

def search_range(D, dt, pixel_size, p_miss):

    # R_max that captures (1 - p_miss) of all Brownian single-frame displacements.
    # Based on the Rayleigh CDF inversion: R_max = sqrt(-4 D dt ln(p_miss))
    
    return np.sqrt(-4 * D * dt * np.log(p_miss)) / pixel_size  # pixels

def check_density(f_frame, SEARCH_RANGE):
    """
    f_frame : trackpy feature DataFrame for a single frame
    SEARCH_RANGE: your computed search range in pixels
    """
    coords = f_frame[['x', 'y']].values
    tree   = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)   # k=2: skip self (k=1)
    d_nn   = np.median(dists[:, 1])      # median nearest-neighbour distance

    print(f"Half-distance between particles = {d_nn/2:.3f} px")

    if SEARCH_RANGE > d_nn / 2:
        print(f"  WARNING: SEARCH_RANGE exceeds d_nn/2 — trajectory swapping likely.")
        print(f"  Either reduce particle concentration or accept a higher p_miss.")
    else:
        print(f"  OK: search circles do not overlap.")