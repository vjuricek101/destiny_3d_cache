#!/usr/bin/env python3
import os
import sys
import train_model

def main():
    parser = train_model.parse_args(args=None)
    # Customize defaults for architectural sweep
    if parser.tech == "ALL":
        default_data = "pareto/SRAM_arch/SRAM_arch_full_data.csv" # Default to SRAM_arch if ALL not aggregated
    else:
        default_data = f"pareto/{parser.tech}_arch/{parser.tech}_arch_full_data.csv"
        
    # We allow the user to override via CLI, but set a smart default
    if not os.path.exists(parser.data) or parser.data == "pareto/full_data.csv":
        parser.data = default_data
        
    if parser.output_dir == "model_output":
        parser.output_dir = f"model_output/{parser.tech.lower()}_arch"

    print(f"--- Training Architectural Surrogate Model ---")
    print(f"Tech: {parser.tech}")
    print(f"Data: {parser.data}")
    print(f"Out:  {parser.output_dir}")
    
    if not os.path.exists(parser.data):
        print(f"ERROR: Architectural data not found at {parser.data}")
        sys.exit(1)
        
    train_model.train(parser)

if __name__ == "__main__":
    main()
