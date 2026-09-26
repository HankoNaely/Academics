import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

x = np.linspace(0., 1., 100)

def  exact_solution(t):
    return np.tan((t**2)/2 + np.pi/4)

def derivative_ode(t, x):
    return t*(x**2 +1)

def euler(h, xi=1., ti=0., tf=1.):

    t = np.arange(ti, tf+h, h) 
    x = np.copy(t)

    x[0] = xi
    t[0] = ti
    
    for i in range(1, len(t)):
        x[i] = x[i-1] + h*derivative_ode(t[i-1], x[i-1])
        
    return t, x

def rk4(h, xi=1., ti=0., tf=1.):
    
    t = np.arange(ti, tf+h, h) 
    x = np.copy(t)

    x[0] = xi
    t[0] = ti
    
    for i in range(1, len(t)):
        k1 = h*derivative_ode(t[i-1], x[i-1])
        k2 = h*derivative_ode(t[i-1] + h/2, x[i-1] + k1/2)
        k3 = h*derivative_ode(t[i-1] + h/2, x[i-1] + k2/2)
        k4 = h*derivative_ode(t[i-1] + h, x[i-1] + k3)
        x[i] = x[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
        
    return t, x

def compute_epsilons(func):
    
    log_space = np.logspace(-5, -1, 13) 
    epsilons = np.empty_like(log_space)
    
    for i, h in np.ndenumerate(log_space):
        t, x = func(h)
        epsilons[i] = np.abs((exact_solution(t[-1])-x[-1])/exact_solution(t[-1]))*100
    
    slope, intercept, r, p, se = linregress(np.log(epsilons), np.log(log_space))
    
    return epsilons, log_space, slope



# FUNCTIONS CALLS
t_euler, x_euler = euler(1e-2)
t_rk4, x_rk4 = rk4(1e-2)
epsilons, hs, slope = compute_epsilons(euler)
epsilons_rk4, hs_rk4, slope_rk4 = compute_epsilons(rk4)           

# PRINTS
'''print(x_euler[-1], t_euler[-1], exact_solution(1.0))
print(f"Epsilon = {round(abs((exact_solution(1.0)-x_euler[-1])/exact_solution(1.0))*100, 3)}%")
print(epsilons, hs)
'''

# PLOTS
fig, ax = plt.subplots()
exact_sol = ax.plot(x, exact_solution(x), '-', color='green', label='Exact solution')
euler_sol = ax.plot(t_euler, x_euler, '.', color='red', label='EFM')
rk4_sol = ax.plot(t_rk4, x_rk4, '+', color='blue', label='RK4') 
plt.legend()
plt.show()

fig, ax = plt.subplots()
epsilon_graph = ax.loglog(hs, epsilons, '-', color='red', label='Euler')
epsilon_graph_rk4 = ax.loglog(hs_rk4, epsilons_rk4, '-', color='blue', label='RK4')
ax.set_xlabel(f'$\log(h)$')
ax.set_ylabel(r'$\epsilon^{(h)}$')
plt.grid()
plt.show()














