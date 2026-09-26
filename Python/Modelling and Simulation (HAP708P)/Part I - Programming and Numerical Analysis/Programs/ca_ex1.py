import numpy as np
import time


def rule_to_binlist_vec(r, order):

    # Prepping arrays of the right dimensions
    k = order
    bin_amount = 2**k 

    # empty array for faster computation since it will get overwritten anyway
    blist = np.empty(bin_amount, dtype=int)
    idx = np.arange(bin_amount, dtype=int)

    # for every value in idx, shift x times r's bits to the right 
    blist = np.bitwise_right_shift(r, idx) 
    
    # compare bitwise to 1, if the far right bit is 1, returns, if its 0 returns 0
    # basically it gets rid of every remaining bits on the left, keeping only the far right one
    blist = np.bitwise_and(blist, 1) 

    return blist


def rule_to_binlist_loop(r, order):

    k = order 
    bin_amount = 2**k

    blist = [0]*bin_amount
    q = r

    for i in range(bin_amount):
        blist[i] =  q % 2
        q //= 2

    return blist

'''
# Benchmark Execution
r_val = 126
order_val = 10
iterations = 100_000

# Time Vectorized Version
start = time.perf_counter()
for i in range(iterations):
    i = rule_to_binlist_vec(r_val, order_val)
time_vec = time.perf_counter() - start

# Time Loop Version
start = time.perf_counter()
for i in range(iterations):
    i = rule_to_binlist_loop(r_val, order_val)
time_loop = time.perf_counter() - start

# Output
print(f"Vectorized Output: {rule_to_binlist_vec(r_val, order_val)}")
print(f"Loop Output:       {rule_to_binlist_loop(r_val, order_val)}")
print(f"Time Vectorized ({iterations:,} iterations): {time_vec:.4f}s")
print(f"Time Loop       ({iterations:,} iterations): {time_loop:.4f}s")
'''