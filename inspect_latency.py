import pandas as pd

df = pd.read_csv('pareto/SRAM_arch/SRAM_arch_pareto.csv')
print(df.nlargest(5, 'cache_hit_latency_ns')[['capacity_kb', 'word_width_bits', 'cache_hit_latency_ns', 'associativity', 'data_total_mats']])
