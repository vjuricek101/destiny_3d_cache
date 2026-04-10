import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_individual_reports(csv_path="pareto/training_dataset_pareto.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found. Run analysis first.")
        return

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create base plots directory
    if not os.path.exists("pareto/plots"):
        os.makedirs("pareto/plots")

    technologies = df['memory_technology'].unique()
    sns.set_theme(style="whitegrid")

    for tech in technologies:
        print(f"Generating independent plots for {tech}...")
        tech_df = df[df['memory_technology'] == tech]
        
        # Create tech-specific directory
        tech_dir = f"pareto/plots/{tech}"
        if not os.path.exists(tech_dir):
            os.makedirs(tech_dir)

        # 1. Area vs Latency (Colored by Capacity)
        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=tech_df, x="Cache Hit Latency (ns)", y="Cache Area (mm^2)", 
            hue="capacity_mb", palette="viridis", alpha=0.7, s=100
        )
        plt.yscale('log')
        plt.title(f"{tech}: Area vs. Latency", fontsize=14)
        plt.legend(title="Capacity (MB)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(tech_dir, "area_vs_latency.png"), bbox_inches='tight', dpi=300)
        plt.close()

        # 2. Energy vs Latency (Colored by Capacity)
        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=tech_df, x="Cache Hit Latency (ns)", y="Cache Hit Energy (nJ)", 
            hue="capacity_mb", palette="magma", alpha=0.7, s=100
        )
        plt.xscale('log'); plt.yscale('log')
        plt.title(f"{tech}: Energy vs. Latency", fontsize=14)
        plt.legend(title="Capacity (MB)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(tech_dir, "energy_vs_latency.png"), bbox_inches='tight', dpi=300)
        plt.close()

        # 3. Leakage vs Latency (Colored by Capacity)
        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=tech_df, x="Cache Hit Latency (ns)", y="Cache Leakage Power (mW)", 
            hue="capacity_mb", palette="rocket", alpha=0.7, s=100
        )
        plt.yscale('log')
        plt.title(f"{tech}: Leakage Power vs. Latency", fontsize=14)
        plt.legend(title="Capacity (MB)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(tech_dir, "leakage_vs_latency.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # 4. Global Comparison Plot
    plt.figure(figsize=(10, 6))
    best_lat = df.groupby(['memory_technology', 'capacity_mb'])['Cache Hit Latency (ns)'].min().reset_index()
    sns.lineplot(data=best_lat, x="capacity_mb", y="Cache Hit Latency (ns)", hue="memory_technology", marker="o", linewidth=2.5)
    plt.xscale('log', base=2)
    plt.title("Technology Scaling Comparison (Min Latency)", fontsize=14)
    plt.savefig("pareto/plots/comparison_scaling.png", bbox_inches='tight', dpi=300)
    print("Done")

if __name__ == "__main__":
    generate_individual_reports()
