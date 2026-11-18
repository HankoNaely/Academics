import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
'''
angles_deg = np.array([96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196, 206, 216, 226, 236, 246, 256, 266, 276])

intensites_exp = np.array([0.59, 19.32, 74.41, 166.75, 281.09, 396.69, 497.91, 639.91, 665.38, 692.22, 669.81, 590.22, 513.21, 388.06, 258.63, 146.13, 61.22, 12.56, 1.98])
'''
angles_deg = np.array([60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 50, 40, 30, 20, 10])

intensites_exp = np.array([6, 37.9, 176, 359, 502, 532, 435, 260, 94, 6.6, 95, 265, 434, 526, 497])


def loi_de_malus(theta_deg, I0, theta0_deg, I_offset):
    theta_rad = np.radians(2*(theta_deg - theta0_deg))
    return I0 * (np.cos(theta_rad) ** 2) + I_offset


p0 = [np.max(intensites_exp), 0.0, np.min(intensites_exp)]
popt, pcov = curve_fit(loi_de_malus, angles_deg, intensites_exp, p0=p0)

I0_fit, theta0_fit, I_offset_fit = popt
perr = np.sqrt(np.diag(pcov))
dI0, dtheta0, dI_offset = perr

residus = intensites_exp - loi_de_malus(angles_deg, *popt)
ss_res = np.sum(residus ** 2)
ss_tot = np.sum((intensites_exp - np.mean(intensites_exp)) ** 2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(ss_res / len(angles_deg))

print(f"I0       = {I0_fit:.3f} ± {dI0:.3f}")
print(f"theta0   = {theta0_fit:.3f} ± {dtheta0:.3f} °")
print(f"I_offset = {I_offset_fit:.3f} ± {dI_offset:.3f}")
print(f"RMSE     = {rmse:.3f}")
print(f"R²       = {r2:.5f}")

angles_lisses = np.linspace(np.min(angles_deg) - 5, np.max(angles_deg) + 5, 500)
intensites_th = loi_de_malus(angles_lisses, I0_fit, theta0_fit, I_offset_fit)


plt.figure(figsize=(9, 5.5), dpi=100)

plt.scatter(
    angles_deg,
    intensites_exp,
    color="crimson",
    s=60,
    edgecolors="black",
    zorder=3,
    label="Mesures expérimentales",
)

label_theorie = (
    f"Loi de Malus théorique\n"
    f"($I_0={I0_fit:.3f}\\pm{dI0:.3f}$, "
    f"$\\theta_0={theta0_fit:.1f}\\pm{dtheta0:.1f}^\\circ$,\n"
    f"$I_{{off}}={I_offset_fit:.3f}\\pm{dI_offset:.3f}$)"
)
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