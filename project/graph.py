import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })
 
# multiple line plot
plt.plot( 'x', 'y1', df, marker='o', markerfacecolor='blue', markersize=12, color='red', linewidth=4)
plt.plot( 'x', 'y2', df, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'y3', df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
plt.legend()
