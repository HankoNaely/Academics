import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Angles de l'analyseur (°)
angles_deg = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

# Tensions 
intensites_exp = np.array([0.98, 0.91, 0.74, 0.49, 0.26, 0.07, 0.01, 0.08, 0.24, 0.51, 0.76, 0.93, 0.99])


# Malus 
def loi_de_malus(theta_deg, I0, theta0_deg, I_offset):
    theta_rad = np.radians(theta_deg - theta0_deg)
    return I0 * (np.cos(theta_rad) ** 2) + I_offset


# FIT
p0 = [np.max(intensites_exp), 0.0, np.min(intensites_exp)]  # Valeurs initiales
popt, _ = curve_fit(loi_de_malus, angles_deg, intensites_exp, p0=p0)

I0_fit, theta0_fit, I_offset_fit = popt


angles_lisses = np.linspace(np.min(angles_deg) - 5, np.max(angles_deg) + 5, 500)
intensites_th = loi_de_malus(angles_lisses, I0_fit, theta0_fit, I_offset_fit)


# PLOT
plt.figure(figsize=(9, 5.5), dpi=100)

# Points expérimentaux
plt.scatter(
    angles_deg,
    intensites_exp,
    color="crimson",
    s=60,
    edgecolors="black",
    zorder=3,
    label="Mesures expérimentales",
)

# Courbe théorique ajustée
label_theorie = f"Loi de Malus théorique\n($I_0={I0_fit:.3f}$, $\\theta_0={theta0_fit:.1f}^\\circ$, $I_{{off}}={I_offset_fit:.3f}$)"
plt.plot(
    angles_lisses,
    intensites_th,
    color="navy",
    linewidth=2,
    linestyle="--",
    label=label_theorie,
)
plt.title(
    "Vérification expérimentale de la Loi de Malus",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Angle de l'analyseur $\\theta$ (°)", fontsize=11)
plt.ylabel("Tension mesurée (V)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()

plt.show()