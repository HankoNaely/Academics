import matplotlib.pyplot as plt

# Dataset labels
datasets = ["R372", "R409", "R458", "R496", "R554", "R609", "R670", "R907"]

# Windowed D CV (%) values from your table
cv_percent = [1.99, 1.09, 2.69, 2.75, 4.34, 2.17, 7.89, 4.09]

# Create plot
plt.figure(figsize=(10, 5))

plt.plot(datasets, cv_percent, marker='o')
plt.axhline(5, linestyle='--')  # stability threshold (5%)

# Labels and title
plt.xlabel("Dataset")
plt.ylabel("Windowed D CV (%)")

# Annotate values
for i, val in enumerate(cv_percent):
    plt.text(i, val + 0.2, f"{val:.2f}", ha='center')

plt.grid(True)

plt.tight_layout()
plt.show()