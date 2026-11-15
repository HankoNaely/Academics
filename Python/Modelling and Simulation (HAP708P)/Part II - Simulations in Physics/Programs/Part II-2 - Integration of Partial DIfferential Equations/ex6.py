import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import time
from ex3 import jacobi
from ex4 import gauss_seidel
from ex5 import overrelax

# Parameters

epsilon = 1e-5
delta = 1e-2
h = delta**2/4
omega = 1.8

M = int(1/delta)

# Init phi 
phi_prev = np.zeros((M+1, M+1), dtype=float)
phi_prev[0,:] = 1.

x = np.arange(M + 1) * delta
y = 1.0 - np.arange(M + 1) * delta
X, Y = np.meshgrid(x, y)

def phi_analytical(X, Y):
    phi_an = 0
    for m in range(100):
        phi_an += (4/np.pi)*(np.sin((2*m+1)*np.pi*X))*np.sinh((2*m+1)*np.pi*Y)/((2*m+1)*np.sinh((2*m+1)*np.pi))

    return phi_an

phi = phi_analytical(X, Y)
phi_jac, t_jac = jacobi(epsilon, phi_prev.copy())
phi_gauss, t_gauss = gauss_seidel(epsilon, phi_prev.copy())
phi_over, t_over = overrelax(epsilon, omega, phi_prev.copy())

e_jac = np.sum(np.abs(phi - phi_jac))/np.sum(np.abs(phi))
e_gauss = np.sum(np.abs(phi - phi_gauss))/np.sum(np.abs(phi))
e_over = np.sum(np.abs(phi - phi_over))/np.sum(np.abs(phi))

print(f'Jacobi : e = {round(e_jac,4)}, t = {t_jac}')
print(f'Gauss-Seidel : e = {round(e_gauss,4)}, t = {t_gauss}')
print(f'Overrelaxation : e = {round(e_over,4)}, t = {t_over}')