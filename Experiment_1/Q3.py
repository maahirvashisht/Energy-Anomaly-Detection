#Q3) Create a sample data set [50-100 instances]

import pandas as pd
import numpy as np

n=100
data=pd.DataFrame({
    "Modified": np.arange(1,n+1),
    "Dataset size(kb)": np.random.randint(50,101,n),
    "Accuracy": np.round(np.random.uniform(80,100,n),2),
    "Training time(hr)": np.round(np.random.uniform(1,10,n),2),
    "Inference time(sec)": np.round(np.random.uniform(0.1,1.0,n),2)
})

print(data)