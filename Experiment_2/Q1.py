#Q1) Analyze the behavior & relationship among MoCT for: (using numpy & matplotlib)
  # a. Unimodel symmetric distribution (Mean ≈ Median ≈ Mode)
  # b. Right skewed distribution (Mode < Median < Mean)
  # c. Left skewed distribution (Mean < Median < Mode)


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

symmetric = np.random.normal(loc=50, scale=10, size=1000)
right_skewed = np.random.exponential(scale=10, size=1000)
left_skewed = 100 - np.random.exponential(scale=10, size=1000)

def calculate_values(data):
    mean = np.mean(data)
    median = np.median(data)
    counts, bins = np.histogram(data, bins=30)
    max_index = np.argmax(counts)
    mode_value = (bins[max_index] + bins[max_index + 1]) / 2
    return mean, median, mode_value

mean_s,median_s,mode_s= calculate_values(symmetric)
mean_r,median_r,mode_r=calculate_values(right_skewed)
mean_l,median_l,mode_l=calculate_values(left_skewed)

print("Unimodel Symmetric Distribution: ")
print(" Mean: ",mean_s)
print(" Median: ",median_s)
print(" Mode: ",mode_s)

print("\nRight Skewed Distribution: ")
print(" Mean: ",mean_r)
print(" Median: ",median_r)
print(" Mode: ",mode_r)

print("\nLeft Skewed Distribution: ")
print(" Mean: ",mean_l)
print(" Median: ",median_l)
print(" Mode: ",mode_l)


plt.figure(figsize=(8, 5))

plt.hist(symmetric, bins=30, alpha=0.7)

plt.axvline(mean_s, linestyle='--', label='Mean')
plt.axvline(median_s, linestyle='-', label='Median')
plt.axvline(mode_s, linestyle=':', label='Mode')

plt.title("Unimodal Symmetric Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()



plt.figure(figsize=(8, 5))

plt.hist(right_skewed, bins=30, alpha=0.7)

plt.axvline(mean_r, linestyle='--', label='Mean')
plt.axvline(median_r, linestyle='-', label='Median')
plt.axvline(mode_r, linestyle=':', label='Mode')

plt.title("Right-Skewed Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()



plt.figure(figsize=(8, 5))

plt.hist(left_skewed, bins=30, alpha=0.7)

plt.axvline(mean_l, linestyle='--', label='Mean')
plt.axvline(median_l, linestyle='-', label='Median')
plt.axvline(mode_l, linestyle=':', label='Mode')

plt.title("Left-Skewed Distribution")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()