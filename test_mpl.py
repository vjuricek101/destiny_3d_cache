import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

df = pd.DataFrame({
    'x': np.random.rand(4738),
    'y': np.random.rand(4738),
    'c': np.random.choice([2.0, 4.0], 4738),
    'type': ['HP']*2369 + ['LOP']*2369
})

x = df['x'].values
y = df['y'].values
sub_col = 'type'
row_st = 'HP'
st_mask = (df[sub_col] == row_st).values

fig, ax = plt.subplots()
try:
    ax.scatter(x[st_mask], y[st_mask], c=df['c'].values[st_mask])
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
