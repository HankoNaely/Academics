"""
Signal-to-Noise Ratio Analysis — Brownian Motion Tracking Results
=================================================================
Savin & Doyle (2005) apparent MSD model:
    MSD_apparent(τ) = 4·D·τ  +  4·ε²  −  4·D·t_exp/3

    Signal  = 4·D·τ                   (pure diffusive term)
    Offset  = 4·ε² − 4·D·t_exp/3     (net bias; can be negative if blur dominates)

    SNR = D·τ / (ε² − D·t_exp/3)     [dimensionless]

If SNR < 0  →  dynamic blur dominates over localization noise:
               the net MSD offset is negative (apparent D is suppressed).
If SNR > 0  →  localization noise dominates: apparent D is inflated.

All quantities in µm and seconds.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data from the results table ───────────────────────────────────────────────
datasets = ["R372", "R409", "R458", "R496", "R554", "R609", "R670", "R907"]

# Experimental diffusion coefficients (µm²/s)  ± uncertainty
D_exp = np.array([0.128, 0.1196, 0.1127, 0.1252, 0.1214, 0.1085, 0.1021, 0.08891])
D_err = np.array([2.9e-5, 2.76e-4, 9.5e-5, 8.9e-5, 8.0e-5, 5.6e-5, 1.39e-4, 7.5e-5])

# Localization error ε  (nm → µm)
epsilon_nm = np.array([35.81, 35.93, 35.09, 34.28, 30.24, 31.19, 30.15, 37.31])
epsilon_um = epsilon_nm / 1000.0          # µm

# Effective frame interval τ  (s)
tau = np.array([0.0672, 0.0672, 0.0672, 0.0672, 0.0672, 0.0672, 0.06735, 0.0672])

# Camera exposure time  (s) — same for all datasets
t_exp = 0.067

# Temperature (°C) — for colour-coding
temperature = np.array([46.3, 44.35, 42.04, 40.42, 38.2, 36.32, 34.43, 28.56])

# ── Compute noise terms and SNR ───────────────────────────────────────────────
loc_noise   =  epsilon_um**2                 # +ε²         (µm²)
dyn_noise   =  D_exp * t_exp / 3            #  D·t_exp/3  (µm²)

signal      = D_exp * tau                   #  D·τ        (µm²)
noise_net   = loc_noise - dyn_noise         #  ε² − D·t_exp/3

SNR         = signal / noise_net            # negative when blur dominates

# ── Propagate uncertainty on SNR from D_err ───────────────────────────────────
#   dSNR/dD = τ·ε² / (ε² − D·t_exp/3)²
dSNR_dD   = tau * loc_noise / noise_net**2
SNR_err   = abs(dSNR_dD * D_err)

# ── Print table ───────────────────────────────────────────────────────────────
header = (f"{'Dataset':<8} | {'D_exp':>10} | {'ε (nm)':>7} | "
          f"{'ε² (pm²)':>10} | {'Dt/3 (pm²)':>10} | "
          f"{'net offset':>12} | {'SNR':>9}")
print("\n" + "=" * len(header))
print("  SNR = D·τ / (ε² − D·t_exp/3)   [Savin & Doyle 2005]")
print("=" * len(header))
print(header)
print("-" * len(header))

for i, ds in enumerate(datasets):
    net_sign = "+" if noise_net[i] >= 0 else "−"
    print(f"{ds:<8} | {D_exp[i]:>10.5f} | {epsilon_nm[i]:>7.2f} | "
          f"{loc_noise[i]*1e6:>10.3f} | {dyn_noise[i]*1e6:>10.3f} | "
          f"  {net_sign}{abs(noise_net[i])*1e6:>8.3f} pm² | {SNR[i]:>+9.3f}")

print("-" * len(header))
print("  SNR < 0 → dynamic blur dominates (net MSD offset is negative, D suppressed)")
print("=" * len(header))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

cmap   = plt.cm.coolwarm_r
norm_c = plt.Normalize(temperature.min(), temperature.max())
colors = cmap(norm_c(temperature))
x = np.arange(len(datasets))

# ── Left panel: signed SNR bar chart ─────────────────────────────────────────
ax1 = axes[0]
bar_colors = ['tomato' if s < 0 else 'steelblue' for s in SNR]
bars = ax1.bar(x, SNR, color=bar_colors, edgecolor='k', linewidth=0.7,
               width=0.6, zorder=3)
ax1.errorbar(x, SNR, yerr=SNR_err, fmt='none', color='k',
             capsize=4, linewidth=1.3, zorder=4)

for i, (bar, snr, err) in enumerate(zip(bars, SNR, SNR_err)):
    va   = 'bottom' if snr >= 0 else 'top'
    ypos = snr + np.sign(snr) * (err + abs(max(SNR) - min(SNR)) * 0.03)
    ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{snr:+.2f}", ha='center', va=va, fontsize=8.5)

ax1.axhline(0, color='black', lw=1.0, ls='-')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=30, ha='right')
ax1.set_ylabel(r"SNR  $= D\tau\,/\,(\varepsilon^2 - Dt_\mathrm{exp}/3)$", fontsize=10)
ax1.grid(axis='y', alpha=0.35, zorder=0)

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor='tomato',    label='SNR < 0  (blur dominates)'),
                Patch(facecolor='steelblue', label='SNR > 0  (loc. noise dominates)')]
ax1.legend(handles=legend_elems, fontsize=9)

# ── Right panel: ε² vs D·t_exp/3 side-by-side, absolute µm² ─────────────────
ax2 = axes[1]
w = 0.35
bars_loc = ax2.bar(x - w/2, loc_noise * 1e6, width=w,
                   label=r'Loc. noise  $\varepsilon^2$  [+]',
                   color='steelblue', alpha=0.85, edgecolor='k', linewidth=0.7)
bars_dyn = ax2.bar(x + w/2, dyn_noise * 1e6, width=w,
                   label=r'Dynamic blur  $Dt_\mathrm{exp}/3$  [−]',
                   color='tomato', alpha=0.85, edgecolor='k', linewidth=0.7)

for i in range(len(datasets)):
    ax2.text(i - w/2, loc_noise[i]*1e6 + 8, f"{loc_noise[i]*1e6:.0f}",
             ha='center', va='bottom', fontsize=7.5)
    ax2.text(i + w/2, dyn_noise[i]*1e6 + 8, f"{dyn_noise[i]*1e6:.0f}",
             ha='center', va='bottom', fontsize=7.5)

ax2.set_xticks(x)
ax2.set_xticklabels(datasets, rotation=30, ha='right')
ax2.set_ylabel(r"Magnitude  (pm²  =  $10^{-6}$ µm²)", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.35, zorder=0)

plt.tight_layout()
# This saves the file in the same folder as your script
plt.savefig("snr_analysis.png", dpi=150, bbox_inches='tight')
print("\nFigure saved → snr_analysis.png")
plt.show()