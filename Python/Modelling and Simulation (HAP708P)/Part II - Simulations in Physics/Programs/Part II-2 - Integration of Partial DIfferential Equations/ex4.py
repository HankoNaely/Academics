import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import time

# Parameters

epsilon = 1e-5
delta = 1e-2
h = delta**2/4

M = int(1/delta)

# Init phi 
phi_prev = np.zeros((M+1, M+1), dtype=float)
phi_prev[0,:] = 1.

def gauss_seidel(epsilon, phi_prev):
    # First diff computed to start the while
    diff = np.inf
    counter = 0
    t_start = time.process_time()


    while diff > epsilon:

        phi = np.copy(phi_prev)

        for k in range(1, M):
            for j in range(1, M):
                phi[k, j] = 0.25*(phi[k+1, j] + phi[k-1, j] + phi[k, j+1] + phi[k, j-1])

        diff = np.sum(np.abs(phi - phi_prev)) / np.sum(np.abs(phi_prev))

        # Swapping references, phi will be overwritten next loop anyway and it prevents accidentally linking both variables to the same array memory
        phi_prev, phi = phi, phi_prev

        counter += 1

    phi = phi_prev

    elapsed = time.process_time() - t_start
    print(f"Done, converged in {counter} iterations ({round(elapsed,3)} s)!")
    return phi, elapsed

if __name__ == "__main__":

    phi, _ = gauss_seidel(epsilon, phi_prev)

    plt.imshow(phi, origin="upper", extent=[0, 1, 0, 1], cmap="inferno")
    plt.colorbar(label=r"$\phi$ (V)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Electrostatic potential (Gauss-Seidel method)")
    plt.show()