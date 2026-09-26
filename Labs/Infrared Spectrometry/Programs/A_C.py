"""
Calibration Beer-Lambert : Aire (A) = (epsilon * l) * C + b
------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ---------------------------------------------------------------
#  DATA
# ---------------------------------------------------------------

C = [0.52, 0.2, 0.66, 0.11]      # concentration connues en argile (%)
A = [26.357, 11.032, 25.046, 7.729]  # aires intégrées correspondante

# ---------------------------------------------------------------
# LINREGRESS
# ---------------------------------------------------------------

C = np.array(C)
A = np.array(A)

result = linregress(C, A)
slope = result.slope          # pente = epsilon * l
intercept = result.intercept  # ordonnée à l'origine
r_value = result.rvalue       # coefficient de corrélation
stderr = result.stderr        # incertitude sur la pente
intercept_stderr = result.intercept_stderr  # incertitude sur l'ordonnée

print("=== Résultats de la régression linéaire ===")
print(f"Pente (epsilon * l)      : {slope:.5g}  ± {stderr:.5g}")
print(f"Ordonnée à l'origine (b) : {intercept:.5g}  ± {intercept_stderr:.5g}")
print(f"Coefficient R^2          : {r_value**2:.5f}")

# ---------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------

C_fit = np.linspace(min(C), max(C), 100)
A_fit = slope * C_fit + intercept

plt.figure(figsize=(7, 5))
plt.scatter(C, A, color="tab:blue", label="Points expérimentaux", zorder=3)
plt.plot(C_fit, A_fit, color="tab:red",
         label=f"Régression : A = {slope:.4g}·C + {intercept:.4g}\n$R^2$ = {r_value**2:.4f}")

plt.xlabel("Concentration C (%)")
plt.ylabel("Aire intégrée A")
plt.title("Droite d'étalonnage Beer-Lambert")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=150)
plt.show()



'''if A_unknown is not None:
    C_unknown = (A_unknown - intercept) / slope
    print(f"\nAire mesurée = {A_unknown} -> Concentration déduite = {C_unknown:.4g}")'''