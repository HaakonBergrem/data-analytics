import pandas as pd
from pathlib import Path


# Creates a new, smaller file with random rows from the original
INPUT_FILE = Path("data/enheter.csv")          
OUTPUT_FILE = Path("data/enheter_150k.csv")    
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