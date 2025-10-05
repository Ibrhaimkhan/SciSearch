import pandas as pd
import glob
import os

folder_path = r"C:\Users\FireA\Documents\GitHub\SciSearch\Combined_data"

# Get all CSV files in that folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Load and combine
dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Save merged version
combined_df.to_csv(os.path.join(folder_path, "combined_df.csv"), index=False)

print("✅ Combined dataset created with shape:", combined_df.shape)
