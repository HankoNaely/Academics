import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np

# PARAMETERS 
# Physical parameters
T_i = 800+273.15
T_a = 20+273.15

R = 0.1
lmbd = 3
alpha = 20
D = 1.6E-6

# Discretization parameters
M = 100
delta = 1/M

window = (60*60*2/15)*D/R**2  # duration of a plot window in tau instead of scd
h_param = 1e-5 # the one h that I actually set manually, before rounding N and taking a new h back
N = int(window/h_param) # How much steps would h_param yield, rounding it so we have an integer
h = window/N # whats the actual h value to return N as integer, closest to h_param

# Non-dimensionalized parameters
theta = np.full(M+1, T_i - T_a, dtype=float)
x = np.linspace(0, delta*M, M+1)
tau = np.linspace(0, h*N, N+1)
c = alpha*R/lmbd

# FUNCTIONS
def FTCS_step(theta, x, c, delta, h):

    j = x/delta
    fwrd = np.roll(theta, -1)
    fwrd[-1] = fwrd[-3] - 2*c*delta*fwrd[-2]
    bckwrd = np.roll(theta, 1)

    a = (1+1/(2*j))**2
    b = (1-1/(2*j))**2

    theta_new = theta + h/(delta**2)*(a*(fwrd - theta) - b*(theta - bckwrd))
    theta_new[0] = theta[0] + 6*h/(delta**2)*(theta[1] - theta[0])
    
    theta[:] = theta_new


n_window = 15
t = 0


fig, ax = plt.subplots()
ax.plot(x*R, theta + T_a, label='0 min')

for w in range(n_window):
    for n in range(N):
        FTCS_step(theta, x, c, delta, h)

    t += round((window*R**2/D)/60)
    ax.plot(x*R, theta + T_a, label=f'{t} min')

ax.set_xlabel('r')
ax.set_ylabel('T(r,t)')
ax.legend()
plt.grid()
plt.show()





