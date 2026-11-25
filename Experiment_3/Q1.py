#Perform the analysis of MoS(measure of spread(range,variance,standard deviation)) on sample data print sets

import numpy as np

data=[10,12,23,14,11,19,18,17,21,25]
print("\nSample data: ",data)

data_range=max(data)-min(data)
variance=np.var(data)
standardDeviation=np.std(data)

print("\nMeasures of Spread: ")
print(" Data range: ",data_range)
print(" Variance: ",variance)
print(" Standard Deviation: ",standardDeviation)