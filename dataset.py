import pandas as pd
import numpy as np

df = pd.read_csv("pareto/SRAM/SRAM_pareto.csv")

for col in ["Cache Hit Latency (ns)", "Cache Hit Energy (nJ)"]:
    vals = df[col]
    print(f"\n{col}")
    print(f"  min:    {vals.min():.6e}")
    print(f"  max:    {vals.max():.6e}")
    print(f"  mean:   {vals.mean():.6e}")
    print(f"  median: {vals.median():.6e}")
    print(f"  % values < 0.001: {(vals < 0.001).mean()*100:.1f}%")