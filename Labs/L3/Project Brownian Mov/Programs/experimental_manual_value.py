import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iapws import IAPWS95
from pathlib import Path
from thermor_calib import T_from_resistance

# ── Configuration ─────────────────────────────────────────────────────────────
F_NUMBER      = 3
SKIP_LIST     = [907]
k_B           = 1.380649e-23 
R_MANU        = 2.965e-6  # Manufacturer radius in meters
DELTA_T       = 1.75      # Fixed temperature offset
SIGMA_T_FIXED = 0.05      # Additional uncertainty for corrected version

# ── Helper Functions ──────────────────────────────────────────────────────────
def get_sigma_T_base(r_val):
    """Calculates base temperature error: 0.3402376992 * R^(-0.0876)"""
    return 0.3402376992 * (r_val ** -0.0876)

def get_D(T_K, radius):
    """Calculates D in µm²/s using IAPWS95 viscosity."""
    if T_K <= 273.15: return np.nan
    try:
        # P = 0.101325 MPa (1 atm)
        eta = float(IAPWS95(T=T_K, P=0.101325).mu)
        D = (k_B * T_K) / (6.0 * np.pi * eta * radius)
        return D * 1e12 
    except: return np.nan

# ── Data Collection ───────────────────────────────────────────────────────────
raw_entries = [
    {"id": 907, "d_exp": 0.0889, "d_err": 0.0001},
    {"id": 670, "d_exp": 0.1021, "d_err": 0.0009},
    {"id": 609, "d_exp": 0.1085, "d_err": 0.0005},
    {"id": 554, "d_exp": 0.1219, "d_err": 0.0003},
    {"id": 496, "d_exp": 0.1265, "d_err": 0.0008},
    {"id": 458, "d_exp": 0.1096, "d_err": 0.0006},
    {"id": 409, "d_exp": 0.1173, "d_err": 0.0004},
    {"id": 372, "d_exp": 0.1283, "d_err": 0.0007},
]

data_rows = []

for entry in raw_entries:
    r_id_raw = entry["id"]
    if r_id_raw in SKIP_LIST: continue

    # CRITICAL FIX: Convert Kelvin output to Celsius for the rest of the script
    t_raw_k, t_meas_uncert = T_from_resistance(r_id_raw * 10)
    t_meas_c = t_raw_k - 273.15
    
    sigma_t_base = get_sigma_T_base(r_id_raw * 10)
    combined_sigma_t = np.sqrt(t_meas_uncert**2 + sigma_t_base**2)
    
    data_rows.append({
        "ID": f"R{r_id_raw}",
        "T_C": t_meas_c,
        "sigma_T_total": combined_sigma_t,
        "D_exp": entry["d_exp"],
        "D_err": entry["d_err"]
    })

df = pd.DataFrame(data_rows)

# ── Theoretical Calculations & Error Propagation ─────────────────────────────
def compute_theo_and_err(row, offset=0, extra_sigma=0):
    # Here T_C is now correctly ~20-25°C
    t_mid = row["T_C"] - offset + 273.15
    s_t   = row["sigma_T_total"] + extra_sigma
    
    d_mid = get_D(t_mid, R_MANU)
    d_hi  = get_D(t_mid + s_t, R_MANU)
    d_lo  = get_D(t_mid - s_t, R_MANU)
    return d_mid, abs(d_hi - d_lo) / 2

# Uncorrected
df[['D_theo_uncorr', 'D_theo_uncorr_err']] = df.apply(
    lambda r: compute_theo_and_err(r, 0, 0), axis=1, result_type='expand'
)
# Corrected
df[['D_theo_corr', 'D_theo_corr_err']] = df.apply(
    lambda r: compute_theo_and_err(r, DELTA_T, SIGMA_T_FIXED), axis=1, result_type='expand'
)

# ── Results Printing ──────────────────────────────────────────────────────────
print(f"\n{'ID':<8} | {'Temp (°C)':<10} | {'Exp. D':<15} | {'Theo. D Corr':<15} | {'Rel. Err'}")
print("-" * 80)

for _, row in df.iterrows():
    exp_str = f"{row['D_exp']:.4f}"
    theo_str = f"{row['D_theo_corr']:.4f}"
    rel_err = abs(row['D_exp'] - row['D_theo_corr']) / row['D_theo_corr'] * 100
    print(f"{row['ID']:<8} | {row['T_C']:<10.2f} | {exp_str:<15} | {theo_str:<15} | {rel_err:.2f}%")

# ── Weighted Fit & Plotting ──────────────────────────────────────────────────
df["eta_meas"] = df["T_C"].apply(lambda t: float(IAPWS95(T=t+273.15, P=0.101325).mu))
df["T_over_eta"] = (df["T_C"] + 273.15) / df["eta_meas"]

df["T_corr_K"] = df["T_C"] - DELTA_T + 273.15
df["eta_corr"] = df["T_corr_K"].apply(lambda t: float(IAPWS95(T=t, P=0.101325).mu))
df["T_over_eta_corr"] = df["T_corr_K"] / df["eta_corr"]

x = df["T_over_eta"].values
y = df["D_exp"].values * 1e-12 
weights = 1.0 / (df["D_err"].values * 1e-12)**2
slope = np.sum(weights * x * y) / np.sum(weights * x**2)
r_fit = k_B / (6.0 * np.pi * slope)

plt.figure(figsize=(10, 6))
plt.errorbar(df["T_over_eta"], df["D_exp"], yerr=df["D_err"], fmt='o', label="Experimental", color="#2c3e8c")
plt.errorbar(df["T_over_eta_corr"], df["D_theo_corr"], yerr=df["D_theo_corr_err"], fmt='s', label="Theory (Corrected)", color="#b71c1c")

x_range = np.linspace(df["T_over_eta_corr"].min()*0.95, df["T_over_eta"].max()*1.05, 100)
plt.plot(x_range, (slope * x_range)*1e12, 'k--', label=f"Fit r={r_fit*1e6:.3f}µm")

plt.xlabel(r"$T / \eta(T)$")
plt.ylabel(r"Diffusion $D$ [µm$^2$/s]")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()