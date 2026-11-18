import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# PARAMETRES THÉORIQUES 
# Coefficients de la loi de Varshni pour GaAs et ses alliages
alpha = 5.4e-4  # eV/K
beta = 204.0    # K

# Gaps à T = 0 K (en eV)
Eg_GaAs_0K = 1.519

# Puits InGaAs avec x = 0.043
x_In = 0.043
Eg_InGaAs_0K = 1.519 - 1.087 * x_In - 0.460 * x_In * (1.0 - x_In)  

# Énergies de confinement trouvées avec GeoGebra (en eV) : Ec_tot = Ec_e + Ec_h
conf_GaAs_5nm   = (78.3 + 14.7) * 1e-3  
conf_GaAs_8nm   = (42.9 + 6.85) * 1e-3   
conf_GaAs_bulk  = 0.0                    #(massif, pas de confinement)
conf_InGaAs_8nm = (19.8 + 5.01) * 1e-3   

# CALCULS EN FONCTION DE LA TEMPÉRATURE
T = np.linspace(77, 300, 224)

# Loi de Varshni pour les deux matériaux
varshni_shift = (alpha * T**2) / (T + beta)
Eg_GaAs_T = Eg_GaAs_0K - varshni_shift
Eg_InGaAs_T = Eg_InGaAs_0K - varshni_shift

# Énergies d'émission hv_f = Eg(T) + Ec_e + Ec_h (en eV)
E_GaAs_5nm   = Eg_GaAs_T + conf_GaAs_5nm
E_GaAs_8nm   = Eg_GaAs_T + conf_GaAs_8nm
E_GaAs_bulk  = Eg_GaAs_T + conf_GaAs_bulk
E_InGaAs_8nm = Eg_InGaAs_T + conf_InGaAs_8nm

# Longueurs d'onde d'émission lambda = hc / E (en nm)
hc_const = 1239.84193  # eV * nm
lambda_GaAs_5nm   = hc_const / E_GaAs_5nm
lambda_GaAs_8nm   = hc_const / E_GaAs_8nm
lambda_GaAs_bulk  = hc_const / E_GaAs_bulk
lambda_InGaAs_8nm = hc_const / E_InGaAs_8nm


# EXPORT DU TABLEAU DE DONNÉES (CSV)
T_sample = np.arange(77, 301, 20)
if 300 not in T_sample:
    T_sample = np.append(T_sample, 300)

# ou d'avance quand on aura nos mesures éventuellement à des T fixés
# T_array = []
# T_sample = np.array(T_array, dtype=float)

varshni_sample = (alpha * T_sample**2) / (T_sample + beta)
Eg_GaAs_sample = Eg_GaAs_0K - varshni_sample
Eg_InGaAs_sample = Eg_InGaAs_0K - varshni_sample

df = pd.DataFrame({
    'T (K)': T_sample,
    'E_GaAs_5nm (eV)': np.round(Eg_GaAs_sample + conf_GaAs_5nm, 4),
    'lambda_GaAs_5nm (nm)': np.round(hc_const / (Eg_GaAs_sample + conf_GaAs_5nm), 1),
    'E_GaAs_8nm (eV)': np.round(Eg_GaAs_sample + conf_GaAs_8nm, 4),
    'lambda_GaAs_8nm (nm)': np.round(hc_const / (Eg_GaAs_sample + conf_GaAs_8nm), 1),
    'E_GaAs_massif (eV)': np.round(Eg_GaAs_sample + conf_GaAs_bulk, 4),
    'lambda_GaAs_massif (nm)': np.round(hc_const / (Eg_GaAs_sample + conf_GaAs_bulk), 1),
    'E_InGaAs_8nm (eV)': np.round(Eg_InGaAs_sample + conf_InGaAs_8nm, 4),
    'lambda_InGaAs_8nm (nm)': np.round(hc_const / (Eg_InGaAs_sample + conf_InGaAs_8nm), 1)
})

csv_filename = "transitions_photoluminescence.csv"
df.to_csv(csv_filename, index=False, sep=";")
print(f"Tableau de données exporté avec succès : '{csv_filename}'")


# GRAPHIQUES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Couleurs et styles
styles = {
    'GaAs 5 nm': ('#1f77b4', '-'),
    'GaAs 8 nm': ('#ff7f0e', '-'),
    'GaAs massif': ('#2ca02c', '--'),
    'InGaAs 8 nm': ('#d62728', '-')
}

# Graphique 1 : Énergie en fonction de T 
ax1.plot(T, E_GaAs_5nm, label='Puits GaAs 5 nm', color=styles['GaAs 5 nm'][0], linestyle=styles['GaAs 5 nm'][1], lw=2)
ax1.plot(T, E_GaAs_8nm, label='Puits GaAs 8 nm', color=styles['GaAs 8 nm'][0], linestyle=styles['GaAs 8 nm'][1], lw=2)
ax1.plot(T, E_GaAs_bulk, label='GaAs massif', color=styles['GaAs massif'][0], linestyle=styles['GaAs massif'][1], lw=2)
ax1.plot(T, E_InGaAs_8nm, label='Puits InGaAs 8 nm', color=styles['InGaAs 8 nm'][0], linestyle=styles['InGaAs 8 nm'][1], lw=2)

ax1.set_title("Énergie de transition en fonction de la température", fontsize=12, fontweight='bold')
ax1.set_xlabel("Température $T$ (K)", fontsize=11)
ax1.set_ylabel("Énergie du photon $h\\nu_f$ (eV)", fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=10)

# Graphique 2 : Longueur d'onde en fonction de T
ax2.plot(T, lambda_GaAs_5nm, label='Puits GaAs 5 nm', color=styles['GaAs 5 nm'][0], linestyle=styles['GaAs 5 nm'][1], lw=2)
ax2.plot(T, lambda_GaAs_8nm, label='Puits GaAs 8 nm', color=styles['GaAs 8 nm'][0], linestyle=styles['GaAs 8 nm'][1], lw=2)
ax2.plot(T, lambda_GaAs_bulk, label='GaAs massif', color=styles['GaAs massif'][0], linestyle=styles['GaAs massif'][1], lw=2)
ax2.plot(T, lambda_InGaAs_8nm, label='Puits InGaAs 8 nm', color=styles['InGaAs 8 nm'][0], linestyle=styles['InGaAs 8 nm'][1], lw=2)

ax2.set_title("Longueur d'onde d'émission en fonction de la température", fontsize=12, fontweight='bold')
ax2.set_xlabel("Température $T$ (K)", fontsize=11)
ax2.set_ylabel("Longueur d'onde $\\lambda$ (nm)", fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()