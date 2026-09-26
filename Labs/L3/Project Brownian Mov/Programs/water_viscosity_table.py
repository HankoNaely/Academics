# Program to build table and plot of dynamic viscosity of water in our considered temp range 
# Reference : IAPWS R12-08 

import numpy as np
import matplotlib.pyplot as plt
import csv
from iapws import IAPWS95

# -----------------------------
# Temperature table
# -----------------------------
temps_C = np.arange(20, 62, 2)
results = []
P = 0.101325 # MPa (= 1 atm)

for T_C in temps_C:
    T_K = T_C + 273.15
    w = IAPWS95(T=T_K, P = P)
    results.append({
        "T_C"       : T_C,
        "T_K"       : T_K,
        "rho"       : w.rho,      # kg/m^3
        "eta_Pa"    : w.mu,       # Pa.s
        "eta_mPa"   : w.mu * 1e3, # mPa.s 
    })

# -----------------------------
# Print
# -----------------------------
print("-" * 65)
print(f" Dynamic water viscosity | IAPWS R12-08 | P = 1 atm")
print("-" * 65)
print(f"{'T (°C)':>7} | {'T (K)':>7} | {'rho (kg/m^3)':>10} | {'eta (Pa.s)':>12} | {'eta (mPa.s)':>10}")
print("-" * 65)
for r in results:
    print(f"{r['T_C']:>7.0f} | {r['T_K']:>7.2f} | {r['rho']:>10.4f} | {r['eta_Pa']:>12.6e} | {r['eta_mPa']:>10.6f}")
print("=" * 65)
print("Source : iapws (IAPWS-95 for rho value, IAPWS R12-08 for eta value), eta_2 = 1")

# -----------------------------
# CSV export
# -----------------------------
with open("viscosity_water_iapws.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["T_C", "T_K", "rho_kg_m3", "eta_Pa_s", "eta_mPa_s"])
    for r in results:
        writer.writerow([r["T_C"], r["T_K"], round(r["rho"], 4), f"{r['eta_Pa']:.6e}", round(r["eta_mPa"], 6)])
print("\nCSV file exported : viscosity_water_iapws.csv")

# -----------------------------
# Plot
# -----------------------------
T_arr = np.array([r["T_C"] for r in results])
eta_arr = np.array([r["eta_mPa"] for r in results])

T_fine = np.arange(20, 60.1, 0.1)
eta_fine = []
for T_C in T_fine:
    w = IAPWS95(T=T_C + 273.15, P=0.101325)
    eta_fine.append(w.mu * 1e3)

fig, ax = plt.subplots(figsize=(8, 5))
 
ax.plot(T_fine, eta_fine, color="#1f77b4", linewidth=2, label="IAPWS R12-08 (continuous)")
ax.scatter(T_arr, eta_arr, color="#d62728", zorder=5, s=50, label="Tabulated points (increment of 2°C)")
 
ax.set_xlabel("Temperature (°C)", fontsize=13)
ax.set_ylabel("Dynamic viscosity η (mPa·s)", fontsize=13)
ax.set_title("Dynamic viscosity of water — IAPWS R12-08", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
 
 
plt.tight_layout()
plt.savefig("viscosity_water_iapws.png", dpi=150)
plt.show()
print("Exported figure : viscosity_water_iapws.png")