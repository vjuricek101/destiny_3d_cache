import pandas as pd
df = pd.read_csv('pareto/SRAM_arch/SRAM_arch_full_data.csv')
df_32 = df[df['capacity_kb'] == 32768].copy()

# Find the row with max latency
max_lat_row = df_32.loc[df_32['cache_hit_latency_ns'].idxmax()]

# Find the row with min area
min_area_row = df_32.loc[df_32['cache_area_mm2'].idxmin()]

# Find the row with min leakage
min_leak_row = df_32.loc[df_32['cache_leakage_mW'].idxmin()]

print("--- Max Latency Row ---")
print(max_lat_row[['cache_hit_latency_ns', 'cache_write_energy_nJ', 'cache_area_mm2', 'cache_leakage_mW', 'data_total_mats']])

print("\n--- Min Area Row ---")
print(min_area_row[['cache_hit_latency_ns', 'cache_write_energy_nJ', 'cache_area_mm2', 'cache_leakage_mW', 'data_total_mats']])

print("\n--- Min Leakage Row ---")
print(min_leak_row[['cache_hit_latency_ns', 'cache_write_energy_nJ', 'cache_area_mm2', 'cache_leakage_mW', 'data_total_mats']])
