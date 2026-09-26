import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from iapws import IAPWS95
from pathlib import Path

# ── Parameters & Constants ───────────────────────────────────────
k_B = 1.380649e-23  # J/K (CODATA 2018)
R_NOMINAL = 2.965e-6   # 3 µm manufacturer radius
BASE_DIR = Path(__file__).resolve().parent

def D_from_temperature(T_K, T_err_K=0.0, r_m=R_NOMINAL):
    """
    Calculates theoretical D in µm²/s using IAPWS viscosity.
    Also propagates temperature uncertainty into D uncertainty.

    Parameters
    ----------
    T_K     : float — temperature in Kelvin
    T_err_K : float — 1-sigma uncertainty on T_K (default 0, no propagation)
    r_m     : float — particle radius in metres

    Returns
    -------
    D_um2   : float — diffusion coefficient in µm²/s
    D_err   : float — uncertainty on D in µm²/s (0 if T_err_K=0)
    """
    w   = IAPWS95(T=T_K, P=0.101325)
    eta = float(w.mu)                                    # Pa·s

    D_m2s = (k_B * T_K) / (6.0 * np.pi * eta * r_m)
    D_um2 = D_m2s * 1e12                                # µm²/s

    if T_err_K == 0.0:
        return D_um2, 0.0

    # ── Error propagation: dD/dT = D*(1/T - (1/eta)*deta/dT) ────
    dT = 0.01                                            # K — numerical step
    eta_plus  = float(IAPWS95(T=T_K + dT, P=0.101325).mu)
    eta_minus = float(IAPWS95(T=T_K - dT, P=0.101325).mu)
    deta_dT   = (eta_plus - eta_minus) / (2 * dT)       # Pa·s/K

    dD_dT  = D_um2 * (1.0 / T_K - deta_dT / eta)       # µm²/s per K
    D_err  = abs(dD_dT) * T_err_K                       # µm²/s

    return D_um2, D_err


def generate_theoretical_data():
    temps_C = np.arange(20, 62, 2)
    data = []
    for T_C in temps_C:
        T_K = T_C + 273.15
        w   = IAPWS95(T=T_K, P=0.101325)
        eta = float(w.mu)
        D_um2, _ = D_from_temperature(T_K)              # no uncertainty needed here

        data.append({
            "T_C":      T_C,
            "T_K":      round(T_K, 2),
            "eta_Pa_s": f"{eta:.6e}",
            "D_um2_s":  f"{D_um2:.6e}"
        })
    return data


if __name__ == "__main__":
    print(f"Generating theoretical baseline for r = {R_NOMINAL*1e6} µm...")
    results = generate_theoretical_data()

    csv_path = BASE_DIR / "diffusion_theoretical.csv"
    fieldnames = ["T_C", "T_K", "eta_Pa_s", "D_um2_s"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Theory file saved to: {csv_path}")