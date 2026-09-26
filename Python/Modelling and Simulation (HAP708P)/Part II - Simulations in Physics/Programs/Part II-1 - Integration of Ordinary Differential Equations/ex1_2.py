import matplotlib.pyplot as plt
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

def compute_epsilons():
    
    log_space = np.logspace(-5, -1, 13) 
    epsilons = np.empty_like(log_space)
    print(log_space)
    
    for i, h in np.ndenumerate(log_space):
        t, x = euler(h)
        print(log_space[i])
        epsilons[i] = np.abs((exact_solution(t[-1])-x[-1])/exact_solution(t[-1]))*100
    return epsilons, log_space

# FUNCTIONS CALLS
t_euler, x_euler = euler(1e-2)
epsilons, hs = compute_epsilons()           

# PRINTS
'''print(x_euler[-1], t_euler[-1], exact_solution(1.0))
print(f"Epsilon = {round(abs((exact_solution(1.0)-x_euler[-1])/exact_solution(1.0))*100, 3)}%")
print(epsilons, hs)
'''
# PLOTS
'''fig, ax = plt.subplots()
exact_sol = ax.plot(x, exact_solution(x), '-', color='green', label='Exact solution')
euler_forward = ax.plot(t_euler, x_euler, '.', color='blue', label='EFM')
plt.legend()
plt.show()'''

fig, ax = plt.subplots()
epsilon_graph = ax.loglog(hs, epsilons, '-')
ax.set_xlabel(f'$\log(h)$')
ax.set_ylabel(r'$\epsilon^{(h)}$')
plt.grid()
plt.show()














