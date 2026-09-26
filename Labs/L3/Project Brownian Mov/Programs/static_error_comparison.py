import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Données extraites de votre tableau
videos = ['R372', 'R409', 'R458', 'R496', 'R554', 'R609', 'R670', 'R907']
epsilon = [35.81, 35.93, 35.09, 34.28, 30.24, 31.19, 30.15, 37.31]
static_threshold = [101.8, 76.5, 68.3, 202.6, 58.9, 57.7, 70.6, 162.5]

df = pd.DataFrame({
    'Video': videos,
    'Epsilon': epsilon,
    'Static_Threshold': static_threshold
})

x = np.arange(len(videos))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, df['Static_Threshold'], width, label='Loc. Error $\epsilon$ (nm)', color='skyblue')
rects2 = ax.bar(x + width/2, df['Epsilon'], width, label='S. Threshold $\hat{\epsilon}_{static}$ (nm)', color='salmon')

ax.set_ylabel('nm')
ax.set_xticks(x)
ax.set_xticklabels(videos)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Epsilon'], df['Static_Threshold'], color='purple', s=100, edgecolors='black')

# Ajout des labels pour chaque point
for i, txt in enumerate(videos):
    plt.annotate(txt, (df['Epsilon'][i]+2, df['Static_Threshold'][i]+0.2))

plt.xlabel('Loc. Error $\epsilon$ (nm)')
plt.ylabel('S. Threshold $\hat{\epsilon}_{static}$ (nm)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, 210)
plt.ylim(0, 210)

plt.tight_layout()
plt.show()