# Enhetsregister Analysis

Small exploratory analysis of Norwegian company data from **Brønnøysundregistrene**.

This project looks at a random sample of **150,000 companies** from the dataset *Hovedenheter fra Enhetsregisteret og Foretaksregisteret*.

## What it does

- Loads a subset of relevant columns from the raw CSV
- Cleans and prepares the data (dates, booleans, missing values)
- Creates a few simple features (e.g. active status, founded year)
- Visualises key patterns:
  - Top municipalities
  - Years with the most new companies
  - Most common organisation forms
  - Top industries (NACE)
  - Employee distribution (companies with 1–100 employees)

## Data source

- **Dataset:** Hovedenheter fra Enhetsregisteret og Foretaksregisteret  
- **Provider:** Brønnøysundregistrene  
- **Download:** [data.brreg.no](https://data.brreg.no/enhetsregisteret/api/enheter/lastned/csv)

The full dataset contains over 1 million rows. This project uses a random sample of 150,000 rows to keep the analysis lightweight.

> Note: The data files are not included in this repository.

## How to run

- Install the required dependencies: pip install -r requirements.txt
- Place the CSV file in the "data" folder
- Run the "data_reduction" file
- Run the "main.py" file

## Project structure

```text
01-enhetsregister/
├── data/                  # Local data (not tracked by git)
├── outputs/ 
├── dataset_reduction.py  
├── main.py                # Main analysis script
├── requirements.txt
└── README.md

