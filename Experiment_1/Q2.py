#Q2) Introduce outlier in Sample & analyze the effects of outliers on MoCT

import pandas as pd
import numpy as np
sample = np.random.randint(1, 10, 6)
data=pd.Series(sample)
print("Original sample: ")
print(data)

print("Before outlier: ")
print("Mean: ",data.mean())
print("Median: ", data.median())
print("Mode: ", data.mode())

data.loc[len(data)] = 100

print("Outlier sample: ")
print(data)

print("After outlier: ")
print("Mean: ",data.mean())
print("Median: ", data.median())
print("Mode: ", data.mode())