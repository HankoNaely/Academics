import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np

# PARAMETERS 
# Physical parameters
lmbd = 5e-11
k = 2*np.pi/lmbd
sigma = 1e-10
m = 9.109e-31
hb = 1.05457182e-34
L = 1e-8

# Discretization parameters
delta = 5e-12
M = int(L/delta)

duration = 8e-16
epsilon = 1e-19 
N = int(duration/epsilon) 
frames = np.array([0, 2e-16, 4e-16, 6e-16, 8e-16])

# Non-dimensionalized parameters
psi = np.zeros(M, dtype=float)
x = np.linspace(0, -L/2 + delta*(M-1), M)
tau = np.linspace(0, epsilon*N, N+1)

beta = (hb*epsilon/(4*m*delta**2))*1j

# Crank-Nicolson
I = np.eye(M, M, dtype=complex)
A = (1+2*beta)*I - beta*(np.roll(I, 1, axis=0) + np.roll(I, -1, axis=0))
B = (1-2*beta)*psi + beta*(np.roll(psi, 1) + np.roll(psi, -1))




for n in range(N):
    psi = np.linalg.solve(A, B)

print(psi)






