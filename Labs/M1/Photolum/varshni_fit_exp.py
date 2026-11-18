import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Données
T_exp = np.array([77, 100, 130, 160, 200, 240, 280, 300]) #K
lambda_exp = np.array([774.8, 777.3, 781.5, 787.0, 795.8, 806.1, 814.7, 818.5]) # nm
delta_lambda = 0.5  # incertitude sur lambda

# Conversion + propa des incertitudes
hc = 1239.84193  # eV * nm
E_exp = hc / lambda_exp
sigma_E = (hc / (lambda_exp**2)) * delta_lambda

# Varshni fit
def varshni_model(T, E0, alpha, beta):
    return E0 - (alpha * T**2) / (T + beta)

p0 = [E_exp[0] + 0.01, 5.4e-4, 204.0]
bounds = ([1.0, 1e-5, 50.0], [2.5, 2e-3, 500.0])

popt, pcov = curve_fit(varshni_model, T_exp, E_exp, p0=p0, sigma=sigma_E, absolute_sigma=True, bounds=bounds)
E0_fit, alpha_fit, beta_fit = popt
dE0, dalpha, dbeta = np.sqrt(np.diag(pcov))

# R² et RMSE
E_pred_fit = varshni_model(T_exp, *popt)

# Coefficient de détermination R²
SS_res = np.sum((E_exp - E_pred_fit)**2)
SS_tot = np.sum((E_exp - np.mean(E_exp))**2)
R2 = 1.0 - (SS_res / SS_tot)

# RMSE par rapport au Fit (en meV)
rmse_fit_meV = np.sqrt(np.mean((E_exp - E_pred_fit)**2)) * 1000

# RMSE par rapport à la la courbe théorique
# GaAs 5 nm : E0_theo = 1.519 + 0.0930 eV, alpha = 5.4e-4, beta = 204
E_theo_poly = (1.519 + 0.0930) - (5.4e-4 * T_exp**2) / (T_exp + 204.0)
rmse_theo_meV = np.sqrt(np.mean((E_exp - E_theo_poly)**2)) * 1000


print(" RÉSULTATS DU FIT")

print(f"E0    = {E0_fit:.4f} ± {dE0:.4f} eV")
print(f"alpha = {alpha_fit:.2e} ± {dalpha:.2e} eV/K  (Théorie : 5.40e-4)")
print(f"beta  = {beta_fit:.1f} ± {dbeta:.1f} K        (Théorie : 204.0)")
print("-" * 60)
print(f"Coefficient R²              = {R2:.5f}")
print(f"RMSE (Écart quadratique / Fit)     = {rmse_fit_meV:.2f} meV")
print(f"RMSE (Écart quadratique / Théorie) = {rmse_theo_meV:.2f} meV")


# Plot
T_fit = np.linspace(70, 310, 200)
E_fit_curve = varshni_model(T_fit, *popt)
E_theo_curve = (1.519 + 0.0930) - (5.4e-4 * T_fit**2) / (T_fit + 204.0)

fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax_main.errorbar(T_exp, E_exp, yerr=sigma_E, fmt='o', color='#d62728', ecolor='black', elinewidth=1, capsize=3, label='Mesures expérimentales')
ax_main.plot(T_fit, E_fit_curve, color='#1f77b4', lw=2, label=f'Fit Varshni ($\\alpha={alpha_fit*1e4:.2f}\\times 10^{{-4}}$, $\\beta={beta_fit:.0f}$ K)')
ax_main.plot(T_fit, E_theo_curve, '--', color='gray', alpha=0.8, label='Théorie pure du TP')

# Bloc texte
stats_text = (f"$R^2 = {R2:.5f}$\n" f"RMSE (fit) = {rmse_fit_meV:.2f} meV\n" f"RMSE (théorie) = {rmse_theo_meV:.2f} meV")
ax_main.text(0.04, 0.12, stats_text, transform=ax_main.transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))

ax_main.set_ylabel("Énergie $E$ (eV)", fontsize=11)
ax_main.set_title("Vérification expérimentale de la loi de Varshni (Puits GaAs 5 nm)", fontsize=12, fontweight='bold')
ax_main.grid(True, linestyle=':', alpha=0.6)
ax_main.legend(loc='upper right', fontsize=9)

# Résidus
residuals_meV = (E_exp - E_pred_fit) * 1000
ax_res.axhline(0, color='black', linestyle='--', lw=1)
ax_res.errorbar(T_exp, residuals_meV, yerr=sigma_E * 1000, fmt='s', color='#1f77b4', 
                ecolor='black', elinewidth=1, capsize=3)
ax_res.set_xlabel("Température $T$ (K)", fontsize=11)
ax_res.set_ylabel("Résidus (meV)", fontsize=10)
ax_res.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()