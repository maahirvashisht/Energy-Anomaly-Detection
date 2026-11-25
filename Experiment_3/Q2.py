#Analyze the impact of outliers in MoS

import numpy as np

data = np.array([10, 12, 14, 15, 16, 18, 20])

data_with_outlier = np.append(data, 100)

def analyze_spread(data):
    data_range=np.max(data)-np.min(data)
    variance=np.var(data)
    std_dev=np.std(data)

    Q1=np.percentile(data, 25)
    Q3=np.percentile(data, 75)
    IQR=Q3-Q1

    return data_range, variance, std_dev, IQR

original=analyze_spread(data)
outlier=analyze_spread(data_with_outlier)

print("Original Data:")
print(data)

print("\nData with Outlier:")
print(data_with_outlier)

print("\nMeasures of Spread")
print("----------------------------------------")
print("Measure          Without Outlier   With Outlier")
print("----------------------------------------")
print(f"Range            {original[0]:<17.2f}{outlier[0]:.2f}")
print(f"Variance         {original[1]:<17.2f}{outlier[1]:.2f}")
print(f"Standard Deviation {original[2]:<15.2f}{outlier[2]:.2f}")
print(f"IQR              {original[3]:<17.2f}{outlier[3]:.2f}")