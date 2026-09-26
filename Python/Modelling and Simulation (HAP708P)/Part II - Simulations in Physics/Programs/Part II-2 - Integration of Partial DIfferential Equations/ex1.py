from cycler import V
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from scipy.constants import zepto
import scipy.special as spe

# PARAMETERS 
a = 1
m = 1
k = 10

h = 1e-2

xi = 0.
pi = -1.

ti = 0.
tf = 100.

x = np.linspace(-2., 2., 100)

# FUNCTIONS
def potential(x):
    return k/4 * (x-1)**2 * (x+1)**2

def force(x):
    return -10*x*(x**2 - 1)  # -V'(x)

def rk_vec(state, h):
    x, p = state[0], state[1]
    return h*np.array([p, force(x)])

def rk4(h, xi, pi, ti, tf):
    
    t = np.arange(ti, tf+h, h) 
    z = np.empty((len(t), 2))

    z[0, 0] = xi
    z[0, 1] = pi
    t[0] = ti

    for i in range(1, len(t)):
        state = z[i-1]
        k1 = rk_vec(state, h)
        k2 = rk_vec(state + k1/2, h)
        k3 = rk_vec(state + k2/2, h) 
        k4 = rk_vec(state + k3, h) 

        z[i] = z[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
        
    return t, z[:,0], z[:,1]

def verlet(h, xi, pi, ti, tf):

    t = np.arange(ti, tf+h, h) 
    z = np.empty((len(t), 2))

    z[0, 0] = xi
    z[0, 1] = pi 
    t[0] = ti

    for i in range(1, len(t)):
        z[i, 0] = z[i-1, 0] + h*z[i-1, 1] + (h**2)*0.5 * force(z[i-1, 0])
        a = force(z[i, 0])
        z[i, 1] = z[i-1, 1] + (h/2)*(force(z[i-1, 0]) + a)

    return t, z[:,0], z[:,1]

def exact_sol(t):
    period = 2 * (2. / 15.) ** 0.25 * spe.ellipk(1. / (12. - 2. * np.sqrt(30.)))
    n = np.floor(t / period)
    tred = t - n * period
    return np.sign(tred - period / 2.) * np.sqrt((np.sqrt(6./5.) - 1.) * ( 1. / spe.ellipj(120 ** (1. / 4.) * tred,  0.5 + np.sqrt(30.) / 12.)[2] ** 2 - 1.))

def energy(p, x):
    return 0.5*p**2 + potential(x)

# FUNCTION CALLS
rk_t, rk_x, rk_p = rk4(h, xi, pi, ti, tf)
ve_t, ve_x, ve_p =  verlet(h, xi, pi, ti, tf)
ex_x = exact_sol(ve_t)

# QUESTION 7
ci = np.array([[-2., -1., -0.1, 0.1, 1., 2., 1.5, 2.1, 1.5, 2.1], [0., 0., 0., 0., 0., 0., 1., 1.,-1., -1.]])

# PLOTS
'''fig, ax = plt.subplots()
venergy = ax.plot(x, potential(x), '-', color='green', label='V(x)')
ax.set_xlabel('x')
ax.set_ylabel('V(x)')
ax.set_title('Potential with respect to position x')
plt.legend()
plt.show()

fig, ax = plt.subplots()
exa = ax.plot(ve_t, ex_x, '-', color='orange', label='ES')
rk = ax.plot(rk_t, rk_x, '--', color='green', label='RK4')
ve = ax.plot(ve_t, ve_x, '--', color='blue', label='Verlet')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('x = x(t)')
plt.legend()
plt.show()

fig, ax = plt.subplots()
rk = ax.plot(rk_t, energy(rk_p, rk_x), '-', color='green', label='RK4')
ve = ax.plot(ve_t, energy(ve_p, ve_x), '-', color='blue', label='Verlet')
ax.set_xlabel('t')
ax.set_ylabel('E')
ax.set_title('Visualisation of energy conservation over time')
plt.legend()
plt.show()'''

eng = np.empty(np.shape(ci)[1])
vel = np.empty((np.shape(ci)[1],2))

for i in range(np.shape(ci)[1]):
    ve_t, ve_x, ve_p =  verlet(h, ci[1,i], ci[0,i], ti, tf)
    vel[i, 1] = ve_x
    vel[i, 2] = ve_p
    eng[i] = np.average(energy(ve_p, ve_x))

col.normalize(np.min(eng), np.max(eng))


plt.plot(ve_x, ve_p, '-', color='green')
     
plt.xlabel('x')
plt.ylabel('p')
plt.title('Phase portrait of the oscillator in a double-well potential')
plt.grid()
plt.legend()
plt.show()





