# Q2) Analyze the impact of various transformations on MoCT


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)

data = np.random.normal(loc=50, scale=10, size=1000)

def calculate_moct(data):
    mean = np.mean(data)
    median = np.median(data)

    counts, bins = np.histogram(data, bins=30)
    max_index = np.argmax(counts)

    mode = (bins[max_index] + bins[max_index + 1]) / 2

    return mean, median, mode

mean, median, mode = calculate_moct(data)

shift = 10

shifted = data + shift

mean_shift, median_shift, mode_shift = calculate_moct(shifted)

scale = 2

scaled = data * scale

mean_scale, median_scale, mode_scale = calculate_moct(scaled)

mixed = data * scale + shift

mean_mixed, median_mixed, mode_mixed = calculate_moct(mixed)


print("Original Data")
print("Mean   :", mean)
print("Median :", median)
print("Mode   :", mode)

print("\nAfter Shift (X + 10)")
print("Mean   :", mean_shift)
print("Median :", median_shift)
print("Mode   :", mode_shift)

print("\nAfter Scale (2X)")
print("Mean   :", mean_scale)
print("Median :", median_scale)
print("Mode   :", mode_scale)

print("\nAfter Mixed Operation (2X + 10)")
print("Mean   :", mean_mixed)
print("Median :", median_mixed)
print("Mode   :", mode_mixed)


plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, alpha=0.7)
plt.axvline(mean, linestyle="--", label="Mean")
plt.axvline(median, linestyle="-", label="Median")
plt.axvline(mode, linestyle=":", label="Mode")
plt.title("Original Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()




plt.figure(figsize=(8, 5))
plt.hist(shifted, bins=30, alpha=0.7)
plt.axvline(mean_shift, linestyle="--", label="Mean")
plt.axvline(median_shift, linestyle="-", label="Median")
plt.axvline(mode_shift, linestyle=":", label="Mode")
plt.title("Shifted Distribution (X + 10)")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()




plt.figure(figsize=(8, 5))
plt.hist(scaled, bins=30, alpha=0.7)
plt.axvline(mean_scale, linestyle="--", label="Mean")
plt.axvline(median_scale, linestyle="-", label="Median")
plt.axvline(mode_scale, linestyle=":", label="Mode")
plt.title("Scaled Distribution (2X)")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()




plt.figure(figsize=(8, 5))
plt.hist(mixed, bins=30, alpha=0.7)
plt.axvline(mean_mixed, linestyle="--", label="Mean")
plt.axvline(median_mixed, linestyle="-", label="Median")
plt.axvline(mode_mixed, linestyle=":", label="Mode")
plt.title("Mixed Transformation (2X + 10)")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()