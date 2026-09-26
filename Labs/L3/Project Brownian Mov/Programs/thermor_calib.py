# Program to build from the calibration curve given by our tutor a useable relationship between R and T
# Calibration data exctracted through WebPlotDigitizer (automeris.io)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
'''
# -----------------------------
# Data (R in Ohm, T in Kelvin)
# -----------------------------
r_data = np.array([
    823.6938356775406, 1383.4488090636057, 1664.8725802135061, 
    1993.1606649023163, 2346.189300901008, 2769.6333521662855, 
    3311.4276368610426, 3924.5888964881506, 6143.744982009576, 
    10396.503018228317
])

t_data = np.array([
    353.03934104493146, 340.777007939576, 336.93240952749113, 
    332.3884147610693, 329.4295102441345, 325.59740105266286, 
    321.77569954503554, 318.48791221862086, 310.064973683428, 
    298.83091974188943
])

# -----------------------------
# Model : Power Law + Offset
# -----------------------------
def model_func(R, a, b, c):
    return a * np.power(R, b) + c

# -----------------------------
# Fit
# -----------------------------
# p0 is the initial guess for better convergence
initial_guess = [1000, -0.15, 80]
params, covariance = curve_fit(model_func, r_data, t_data, p0=initial_guess)

a_opt, b_opt, c_opt = params

print("--- Fit result ---")
print(f"Model : T = {a_opt:.4f} * R^({b_opt:.4f}) + {c_opt:.4f}")
print("------------------------")

# -----------------------------
# Plot
# -----------------------------
r_range = np.linspace(min(r_data), max(r_data), 100)
t_fit = model_func(r_range, *params)

plt.figure(figsize=(10, 6))
plt.scatter(r_data, t_data, color='red', label='Real data (calibration curve)')
plt.plot(r_range, t_fit, label=f'Fit: {a_opt:.1f}*R^{b_opt:.3f} + {c_opt:.1f}')
plt.xlabel('Resistance (Ohm)')
plt.ylabel('Temperature (K)')
plt.title('Calibration Temperature vs Resistance')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
'''

def T_from_resistance(R):
    """
    Returns (T_K, T_err) using the power law fit and 0.8% uncertainty on R.
    """
    a, b, c = 485.4990, -0.0876, 83.1835
    
    T_K = a * np.power(R, b) + c
    
    # Error propagation: delta_T = |dT/dR| * delta_R
    # Since delta_R = 0.008 * R, the math simplifies to:
    T_err = 0.008 * abs(a * b) * np.power(R, b)
    
    return T_K, T_err