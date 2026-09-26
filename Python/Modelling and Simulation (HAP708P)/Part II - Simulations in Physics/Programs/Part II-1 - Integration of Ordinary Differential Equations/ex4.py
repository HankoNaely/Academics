import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.integrate import solve_ivp
import numpy as np

x = np.linspace(0., 1., 100)

def  exact_solution(t):
    return np.tan((t**2)/2 + np.pi/4)
    
def derivative_ode(t, x):
    return t*(x**2 +1)

def python_solver(xi=1., ti=0., tf=1.):
    ini = np.array([xi])
    sol = solve_ivp(derivative_ode, [ti, tf], ini, method='DOP853')
    
    return sol.t, sol.y[0]
   

# FUNCTIONS CALLS
t_rk8, x_rk8 = python_solver()


epsilon = np.abs((exact_solution(t_rk8[-1])-x_rk8[-1])/exact_solution(t_rk8[-1]))*100


# PRINTS
print(f"Epsilon = {epsilon}%")


# PLOTS
fig, ax = plt.subplots()
exact_sol = ax.plot(x, exact_solution(x), '-', color='green', label='Exact solution')
rk8_sol = ax.plot(t_rk8, x_rk8, '.', color='blue', label='RK8')
plt.legend()
plt.show()















