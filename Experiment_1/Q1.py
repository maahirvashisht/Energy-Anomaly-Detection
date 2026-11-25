#Q1) Evaluate MoCT of random samples

import pandas as pd
import numpy as np
sample = np.random.randint(1, 10, 6)
data=pd.Series(sample)
print("Random sample: ")
print(data)

print("Mean: ",data.mean())
print("Median: ", data.median())
print("Mode: ", data.mode())

