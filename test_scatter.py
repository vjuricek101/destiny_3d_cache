import pandas as pd
import numpy as np

# Create dummy data like the real one
df = pd.DataFrame({
    'cache_hit_latency_ns': np.random.rand(4738),
    'cache_write_energy_nJ': np.random.rand(4738),
    'capacity_kb': np.random.choice([2.0, 4.0], 4738),
    'device_roadmap': ['HP']*2369 + ['LOP']*2369
})

x = df['cache_hit_latency_ns'].values
y = df['cache_write_energy_nJ'].values
sub_col = 'device_roadmap'
row_st = 'HP'
st_mask = (df[sub_col] == row_st).values

print("x shape:", x.shape)
print("st_mask shape:", st_mask.shape)
print("x[st_mask] shape:", x[st_mask].shape)
print("c shape:", df['capacity_kb'].values[st_mask].shape)

