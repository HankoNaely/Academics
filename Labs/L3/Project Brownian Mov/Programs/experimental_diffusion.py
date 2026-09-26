import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iapws import IAPWS95
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
F_NUMBER  = 3
SKIP_LIST = [907]
BASE_DIR  = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "results"
k_B       = 1.380649e-23 
R_MANU    = 2.965e-6  # Manufacturer radius in meters
DELTA_T   = 1.75      # Fixed offset
SIGMA_T_FIXED = 0.05  # Additional uncertainty for corrected version

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
data_rows = []
folder_pattern = re.compile(rf"R(\d+)_T1_F{F_NUMBER}$")

if RESULTS_PATH.exists():
    for folder in sorted(os.listdir(RESULTS_PATH)):
        match = folder_pattern.match(folder)
        if not match or int(match.group(1)) in SKIP_LIST: continue

        txt_file = RESULTS_PATH / folder / "parameters_and_results.txt"
        if not txt_file.exists(): continue

        with open(txt_file, "r", encoding="utf-8") as fh:
            content = fh.read()

        m_temp = re.search(r"Temperature \(T\)\s*:\s*([\d.]+)\s*°C", content)
        m_exp  = re.search(r"Experimental D.*?:\s*([\d.]+)\s*[±]\s*([\d.]+)", content)
        
        if m_temp and m_exp:
            r_id_val = int(match.group(1)) * 10  # R372 -> 3720
            t_meas_c = float(m_temp.group(1))
            
            # Calculate specific T uncertainty for this R
            sigma_t_base = get_sigma_T_base(r_id_val)
            
            data_rows.append({
                "ID": f"R{match.group(1)}",
                "T_C": t_meas_c,
                "sigma_T_base": sigma_t_base,
                "D_exp": float(m_exp.group(1)),
                "D_err": float(m_exp.group(2))
            })

df = pd.DataFrame(data_rows)

# ── Theoretical Calculations & Error Propagation ─────────────────────────────
def compute_theo_and_err(row, offset=0, extra_sigma=0):
    t_mid = row["T_C"] - offset + 273.15
    s_t   = row["sigma_T_base"] + extra_sigma
    
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
print(f"\n{'ID':<8} | {'Exp. D (µm²/s)':<18} | {'Theo. D Corr (µm²/s)':<22} | {'Rel. Error':<10}")
print("-" * 75)

for _, row in df.iterrows():
    exp_str = f"{row['D_exp']:.4f} ± {row['D_err']:.4f}"
    theo_str = f"{row['D_theo_corr']:.4f} ± {row['D_theo_corr_err']:.4f}"
    
    # Relative error calculation (percentage)
    rel_err = abs(row['D_exp'] - row['D_theo_corr']) / row['D_theo_corr'] * 100
    
    print(f"{row['ID']:<8} | {exp_str:<18} | {theo_str:<22} | {rel_err:.2f}%")

# ── Weighted Fit ──────────────────────────────────────────────────────────────
df["eta_meas"] = df["T_C"].apply(lambda t: float(IAPWS95(T=t+273.15, P=0.101325).mu))
df["T_over_eta"] = (df["T_C"] + 273.15) / df["eta_meas"]

x = df["T_over_eta"].values
y = df["D_exp"].values * 1e-12 # to meters
weights = 1.0 / (df["D_err"].values * 1e-12)**2
slope = np.sum(weights * x * y) / np.sum(weights * x**2)
r_fit = k_B / (6.0 * np.pi * slope)

# ── Plotting ──────────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 6))
plt.errorbar(df["T_over_eta"], df["D_exp"], yerr=df["D_err"], fmt='o', 
             label="Experimental $D$", capsize=3, color="#2c3e8c")
plt.errorbar(df["T_over_eta"], df["D_theo_uncorr"], yerr=df["D_theo_uncorr_err"], 
             fmt='', label="Theoretical (T Uncorrected)", alpha=0.4, color="green")
plt.errorbar(df["T_over_eta"], df["D_theo_corr"], yerr=df["D_theo_corr_err"], 
             fmt='', label="Theoretical (T Corrected)", color="#b71c1c")

x_range = np.linspace(x.min()*0.9, x.max()*1.1, 100)
plt.plot(x_range, (slope * x_range)*1e12, 'k--', alpha=0.5, label=f"Fit $r={r_fit*1e6:.3f}$µm")

plt.xlabel(r"$T / \eta(T)$ [K Pa$^{-1}$ s$^{-1}$]")
plt.ylabel(r"Diffusion $D$ [µm$^2$/s]")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()