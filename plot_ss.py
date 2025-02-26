import numpy as np
import matplotlib.pyplot as plt

# File path
file_path = "training.txt"

# Read the file
with open(file_path, "r") as f:
    lines = f.readlines()

# Determine the number of superdroplets
S = len(lines) // 83  # Each superdroplet has 83 lines

# Lists to store supersaturation values
actual_s = []
filtered_s = []

# Extract supersaturation values
for i in range(S):
    base_index = i * 83  # Start index of each superdroplet
    actual_s.append(float(lines[base_index].split()[-1]))  # Supersaturation from line 1
    filtered_s.append(float(lines[base_index + 1].split()[0]))  # Filtered supersaturation from line 2

# Convert lists to numpy arrays
actual_s = np.array(actual_s)
filtered_s = np.array(filtered_s)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(actual_s, filtered_s, alpha=0.6, label="Filtered vs. Actual Supersaturation")
plt.plot([min(actual_s), max(actual_s)], [min(actual_s), max(actual_s)], 'r--', label="y=x (Perfect Match)")
plt.xlabel("Actual Supersaturation (s)")
plt.ylabel("Filtered Supersaturation (s)")
plt.title("Comparison of Filtered and Actual Supersaturation")
plt.legend()
plt.grid(True)
plt.show()

