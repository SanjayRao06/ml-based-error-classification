import pandas as pd
import sys

# Works for both c++ and python datasets.

if len(sys.argv) != 2:
    print("Usage: python clean_dataset.py <dataset.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = input_file.replace(".csv", "_clean.csv")

df = pd.read_csv(input_file)
print(f"Total entries before cleanup: {len(df)}")

df_clean = df.drop_duplicates(subset=["error_message"])
print(f"Total entries after cleanup: {len(df_clean)}")
print(f"Removed {len(df) - len(df_clean)} duplicates\n")

print("Entries per error_type after cleanup:")
print(df_clean["error_type"].value_counts())

df_clean.to_csv(output_file, index=False)
print(f"\n✅ Cleaned dataset saved as '{output_file}'")
