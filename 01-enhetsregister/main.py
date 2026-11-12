# This file analyses a random sample of 150k rows from the "Enhetsregister" dataset. 
# First, a dataframe using only relevant columns is created. Then some simple cleaning and feature creation is done. 
# Lastly, the dataframe is analysed and the results visualized.

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# This ensures the file is accessible regardless of the location of the user
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "enheter_150k.csv"


# Columns to keep
USE_COLS = [
    "organisasjonsnummer",
    "navn",
    "organisasjonsform.kode",
    "organisasjonsform.beskrivelse",
    "naeringskode1.kode",
    "naeringskode1.beskrivelse",
    "antallAnsatte",
    "konkurs",
    "underAvvikling",
    "registreringsdatoenhetsregisteret",
    "stiftelsesdato",
    "forretningsadresse.kommune",
    "forretningsadresse.kommunenummer",
    "forretningsadresse.postnummer",
    "registrertIMvaRegisteret",
    "registrertIForetaksregisteret",
]

# Creates a dataframe from the csv file using the specified columns
def load_data(path: Path) -> pd.DataFrame:
    # Load only the columns we need
    print("Loading data...")
    df = pd.read_csv(
        path,
        usecols=USE_COLS,
        # In order for pandas to read the types correctly
        dtype={
            "organisasjonsnummer": "string",
            "forretningsadresse.kommunenummer": "string",
            "forretningsadresse.postnummer": "string",
            "naeringskode1.kode": "string",
        },
        low_memory=False,
    )
    print(f"Loaded {len(df):,} rows")
    return df

# Cleans and prepares the dataset 
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")

    # Renames columns to cleaner English names
    df = df.rename(columns={
        "organisasjonsnummer": "org_nr",
        "navn": "name",
        "organisasjonsform.kode": "org_form_code",
        "organisasjonsform.beskrivelse": "org_form",
        "naeringskode1.kode": "nace_code",
        "naeringskode1.beskrivelse": "nace_description",
        "antallAnsatte": "employees",
        "konkurs": "is_bankrupt",
        "underAvvikling": "is_under_liquidation",
        "registreringsdatoenhetsregisteret": "registered_date",
        "stiftelsesdato": "founded_date",
        "forretningsadresse.kommune": "municipality",
        "forretningsadresse.kommunenummer": "municipality_code",
        "forretningsadresse.postnummer": "postal_code",
        "registrertIMvaRegisteret": "is_vat_registered",
        "registrertIForetaksregisteret": "is_in_business_register",
    })

    # Converts dates
    date_cols = ["registered_date", "founded_date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Converts boolean-like columns
    bool_cols = ["is_bankrupt", "is_under_liquidation", "is_vat_registered", "is_in_business_register"]
    for col in bool_cols:
        df[col] = df[col].astype("boolean")

    # Ensures employee-values are numeric, and filled with "0" if "NaN" or "None" are present
    df["employees"] = pd.to_numeric(df["employees"], errors="coerce")
    df["employees"] = df["employees"].fillna(0).astype("int32")

    # Creates useful features
    df["has_employees"] = df["employees"] > 0
    df["registration_year"] = df["registered_date"].dt.year
    df["founded_year"] = df["founded_date"].dt.year

    # Simple active/inactive flag that returns true if the company is neither bankrupt nor under liquidation.  
    df["is_active"] = ~(df["is_bankrupt"].fillna(False) | df["is_under_liquidation"].fillna(False))

    # Drops rows with missing critical info
    df = df.dropna(subset=["org_nr", "org_form_code"])

    print(f"After cleaning: {len(df):,} rows")

    return df

# Creates visualizations of important information from the dataset (the illustrations are not stored)
def create_visualizations(df: pd.DataFrame) -> None:
    print("Creating visualizations...")

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # 1. Top 10 municipalities
    top_muni = df["municipality"].value_counts().head(10)
    plt.figure()
    sns.barplot(x=top_muni.values, y=top_muni.index, color="steelblue", legend=False)
    plt.title("Top 10 municipalities by number of companies")
    plt.xlabel("Number of companies")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


    # 2. Top 10 years where the most new companies were founded
    top_years = df["founded_year"].value_counts().head(10)

    # Removes decimals
    top_years.index = top_years.index.astype("Int64").astype(str)

    plt.figure()
    sns.barplot(
        x=top_years.values,
        y=top_years.index,
        color="steelblue"
    )
    plt.title("Top 10 years where the most new companies were founded")
    plt.xlabel("Number of new companies")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.show()


    # 3. Top 10 organisation forms
    top_org = df["org_form"].value_counts().head(10)
    plt.figure()
    sns.barplot(x=top_org.values, y=top_org.index, color="steelblue", legend=False)
    plt.title("Top 10 organisation forms")
    plt.xlabel("Number of companies")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


    # 4. Top 10 industries
    top_nace = df["nace_description"].value_counts().head(10)

    # Collapses the text if it is too long
    def wrap_label(text, width=30):
        text = str(text)
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return "\n".join(lines)

    labels = [wrap_label(label, width=30) for label in top_nace.index]

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=top_nace.values,
        y=labels,
        color="steelblue"
    )
    plt.title("Top 10 industries")
    plt.xlabel("Number of companies")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


    # 5. Employee distribution (includes only companies with 100 or fewer employees since most companies are in this range)
    emp = df.loc[df["employees"] > 0, "employees"]
    emp_filtered = emp[emp <= 100]

    plt.figure(figsize=(12, 6))
    sns.histplot(emp_filtered, bins=50, color="steelblue")

    plt.title("Distribution of employees (1–100 employees)")
    plt.xlabel("Number of employees")
    plt.ylabel("Number of companies")

    plt.xticks(range(0, 101, 5))

    plt.tight_layout()
    plt.show()


# Prints some basic overviews stats
def basic_overview(df: pd.DataFrame) -> None:
    print("\n=== Basic Overview ===")
    print(f"Total companies: {len(df):,}")
    print(f"Active companies: {df['is_active'].sum():,}")
    print(f"Bankrupt: {df['is_bankrupt'].sum():,}")
    print(f"Average number of employees: {df['employees'].mean():.1f}")

def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    create_visualizations(df)
    basic_overview(df)

    print("\nDone!")

if __name__ == "__main__":
    main()