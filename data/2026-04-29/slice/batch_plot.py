import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

"""
this meant to use after runn_all.py is to graph all event slices, better copy to your result folder because absolute path is not consifured for output
"""
def batch_plot(slice_folder):
    folder = Path(slice_folder)
    # Find all CSV files in the folder
    csv_files = list(folder.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {slice_folder}")
        return

    print(f"Found {len(csv_files)} files. Starting plotting...")

    for csv_path in csv_files:
        try:
            # 1. Load Data
            df = pd.read_csv(csv_path)
            df['time_tag'] = pd.to_datetime(df['time_tag'])
            
            # 2. Identify the flux column (it's the one that isn't time_tag)
            flux_col = [c for c in df.columns if c != 'time_tag'][0]
            
            # 3. Create Plot
            plt.figure(figsize=(10, 5))
            plt.plot(df['time_tag'], df[flux_col], color='tab:red', linewidth=1.5)
            
            # Format
            plt.yscale('log')
            plt.title(f"Event Slice: {csv_path.stem}")
            plt.ylabel(f"Flux ({flux_col})")
            plt.xlabel("Time (UTC)")
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # 4. Save next to the CSV
            output_png = csv_path.with_suffix('.png')
            plt.savefig(output_png)
            plt.close() # Memory management: close figure after saving
            
            print(f"✅ Created: {output_png.name}")
            
        except Exception as e:
            print(f"❌ Failed to plot {csv_path.name}: {e}")

if __name__ == "__main__":
    # You can pass the path as a terminal argument
    path_arg = Path('.')
    batch_plot(path_arg)