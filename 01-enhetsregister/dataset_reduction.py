# This file takes a dataset and reduces its size to 150k rows that are randomly selected. The new file is stored in the /data folder

import pandas as pd
from pathlib import Path

# This ensures the file is accessible regardless of the location of the user
BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "data" / "enheter.csv"         
OUTPUT_FILE = BASE_DIR / "data" / "enheter_150k.csv"   
SAMPLE_SIZE = 150_000
RANDOM_SEED = 42                               

def main():
    print("Reading full file (this may take a while)...")
    
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Original size: {len(df):,} rows")

    if len(df) <= SAMPLE_SIZE:
        print("File is already smaller than sample size. Keeping all rows.")
        df_sample = df
    else:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        print(f"Sampled {SAMPLE_SIZE:,} random rows")

    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to: {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()