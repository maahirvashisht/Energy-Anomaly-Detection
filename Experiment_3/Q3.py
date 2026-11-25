# ______ Transformation(x_new=c*x_old+d) on MoS

import numpy as np

data=np.array([10, 12, 14, 15, 16, 18, 20])

c=2
d=5

new_data=c*data+d

def calculate_mos(data):
    Range=np.max(data)-np.min(data)
    Variance=np.var(data)
    Std_Dev=np.std(data)
    IQR=np.percentile(data,75)-np.percentile(data,25)

    return Range,Variance,Std_Dev,IQR

original_mos=calculate_mos(data)
new_mos=calculate_mos(new_data)

print("Original Data:")
print(data)

print("\nTransformed Data:")
print(new_data)

print("\nMeasures of Spread")
print("-----------------------------")
print("Measure       Original    New")
print("-----------------------------")
print(f"Range         {original_mos[0]:.2f}       {new_mos[0]:.2f}")
print(f"Variance      {original_mos[1]:.2f}       {new_mos[1]:.2f}")
print(f"Std Deviation {original_mos[2]:.2f}        {new_mos[2]:.2f}")
print(f"IQR           {original_mos[3]:.2f}        {new_mos[3]:.2f}")