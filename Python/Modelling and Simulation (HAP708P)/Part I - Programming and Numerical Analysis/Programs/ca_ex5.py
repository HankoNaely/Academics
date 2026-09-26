import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from ca_ex4 import apply_rule

def ca_iterate(array, rule, order, niter):
    
    array = np.tile(array, (niter+1, 1))

    for i in range(1, niter+1):
        array[i,:] = apply_rule(array[i-1,:], rule, order)

    return array

output = ca_iterate([0, 0, 1, 0, 0], 110, 3, 10)
print(output)

plt.figure(figsize=(10,10))
plt.imshow(output,origin="upper",cmap="Greys",aspect='auto')
plt.show()