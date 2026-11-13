import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# This ensures the file is accessible regardless of the location of the user
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "enheter_150k.csv"


# Columns to keep
USE_COLS = [
    "organisasjonsnummer",
    "organisasjonsform.kode",
    "organisasjonsform.beskrivelse",
    "naeringskode1.kode",
    "naeringskode1.beskrivelse",
    "antallAnsatte",
    "stiftelsesdato",
    "registreringsdatoenhetsregisteret",
    "forretningsadresse.kommunenummer",
    "registrertIMvaRegisteret",
    "registrertIForetaksregisteret",
    "institusjonellSektorkode.kode",
]

# Creates a dataframe from the csv file using the specified columns
def load_data(path: Path) -> pd.DataFrame:
    # Load only the columns we need
    print("Loading data...")
    df = pd.read_csv(
        path,
        usecols=USE_COLS,
        dtype={
            "organisasjonsnummer": "string",
            "organisasjonsform.kode": "string",
            "naeringskode1.kode": "string",
            "forretningsadresse.kommunenummer": "string",
            "institusjonellSektorkode.kode": "string",
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
        "organisasjonsform.kode": "org_form_code",
        "organisasjonsform.beskrivelse": "org_form",
        "naeringskode1.kode": "nace_code",
        "naeringskode1.beskrivelse": "industry",
        "antallAnsatte": "employees",
        "stiftelsesdato": "founded_date",
        "registreringsdatoenhetsregisteret": "registered_date",
        "forretningsadresse.kommunenummer": "municipality_code",
        "registrertIMvaRegisteret": "is_vat_registered",
        "registrertIForetaksregisteret": "is_in_business_register",
        "institusjonellSektorkode.kode": "sector_code",
    })

    # Converts dates
    df["founded_date"] = pd.to_datetime(df["founded_date"], errors="coerce")
    df["registered_date"] = pd.to_datetime(df["registered_date"], errors="coerce")

    # Converts boolean-like columns
    df["is_vat_registered"] = df["is_vat_registered"].astype("boolean")
    df["is_vat_registered"] = df["is_vat_registered"].fillna(False).astype(int)
    df["is_in_business_register"] = df["is_in_business_register"].astype("boolean")
    df["is_in_business_register"] = df["is_in_business_register"].fillna(False).astype(int)

    # Fills unspecified codes with "unknown"
    df["municipality_code"] = df["municipality_code"].fillna("unknown")
    df["sector_code"] = df["sector_code"].fillna("unknown")
    
    # Reduces municipality cardinality a bit (keeps most common, groups rest)
    top_muni = df["municipality_code"].value_counts().head(40).index
    df["municipality_code"] = df["municipality_code"].where(
        df["municipality_code"].isin(top_muni), "other"
    )

    # Ensures employee-values are numeric, and filled with "0" if "NaN" or "None" are present
    df["employees"] = pd.to_numeric(df["employees"], errors="coerce").fillna(0).astype("int32")
    df["has_employees"] = (df["employees"] > 0).astype(int)

    # Company age in years
    df["company_age"] = (pd.Timestamp.today() - df["founded_date"]).dt.days / 365.25

    # Clean industry text
    df["industry"] = df["industry"].astype("string").str.strip()

    # Drops rows with missing critical values
    df = df.dropna(subset=["org_nr", "industry", "org_form_code"])
    df = df[df["industry"].str.len() > 0]

    # Removes clearly missing/unspecified industries
    invalid_industry = df["industry"].str.lower().isin([
        "uoppgitt",
        "ukjent",
        "unknown",
        "nan",
        "none",
    ])
    df = df[~invalid_industry]

    # Keeps rows with a valid company age
    df = df[df["company_age"].notna() & (df["company_age"] >= 0)]

    print(f"After cleaning: {len(df):,} rows")
    return df

# Keeps only companies in the top N industries
def filter_top_industries(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    print(f"Filtering to top {top_n} industries...")

    top_industries = df["industry"].value_counts().head(top_n).index
    filtered = df[df["industry"].isin(top_industries)].copy()

    print("Top industries:")
    print(filtered["industry"].value_counts())
    print(f"Rows used for modeling: {len(filtered):,}")

    return filtered

# Prepares X (features) and y (target) to later be used in the training 
def prepare_features(df: pd.DataFrame):
    print("Preparing features and target...")

    feature_df = df.copy()

    # Features used by the models
    categorical_features = ["org_form_code", "municipality_code", "sector_code"]
    numeric_features = [
        "employees",
        "has_employees",
        "is_vat_registered",
        "is_in_business_register",
        "company_age",
    ]

    X = feature_df[categorical_features + numeric_features]
    y = feature_df["industry"]

    return X, y, categorical_features, numeric_features

# Splits the dataset 80/20 (training/testing), and trains a baseline for comparison, a logistic regression model, and a random forest model
def train_and_evaluate(X, y, categorical_features, numeric_features):
    print("\nSplitting into train (80%) and test (20%)...")

    # stratify=y keeps the same industry mix in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {len(X_train):,}")
    print(f"Test size:  {len(X_test):,}")

    # One-hot encodes categorical columns (0s and 1s); keeps numeric columns as they are
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    # Baseline: always predicts the most common industry in the training set
    majority_class = y_train.mode().iloc[0]
    baseline_pred = np.full(shape=len(y_test), fill_value=majority_class)
   
    results = [
        {
            "model": "Baseline",
            "accuracy": accuracy_score(y_test, baseline_pred),
            "macro_f1": f1_score(y_test, baseline_pred, average="macro"),
        }
    ]

    # Model 1: Logistic Regression 
    print("Training Logistic Regression...")
    log_reg = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)

    results.append({
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, log_pred),
        "macro_f1": f1_score(y_test, log_pred, average="macro"),
    })

    # Model 2: Random Forest
    print("Training Random Forest...")
    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results.append({
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, rf_pred),
        "macro_f1": f1_score(y_test, rf_pred, average="macro"),
    })

    results_df = pd.DataFrame(results)
    print("\nTest set results:")
    print(results_df.round(3).to_string(index=False))

    return results_df, rf, y_train

# Plots the number of companies in the top 10 industries
def plot_industry_distribution(df: pd.DataFrame) -> None:
    counts = df["industry"].value_counts()

    # Wraps long labels so they fit better
    def wrap_label(text, width=28):
        words = str(text).split()
        lines, current = [], ""
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

    labels = [wrap_label(i) for i in counts.index]

    plt.figure(figsize=(12, 7))
    sns.barplot(x=counts.values, y=labels, color="steelblue")
    plt.title("Top 10 industries used as prediction target")
    plt.xlabel("Number of companies")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# Compares models on accuracy of predicting the industry a company belongs to
def plot_model_comparison(results_df: pd.DataFrame) -> None:
    plot_df = results_df.melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.title("Model comparison on test set")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

# Plots simple feature importance from the Random Forest
def plot_feature_importance(rf_pipeline, categorical_features, numeric_features) -> None:
    # Because categorical features are one-hot encoded, they have to be connverted back to their original feature names 
    ohe = rf_pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = ohe_names + numeric_features

    importances = rf_pipeline.named_steps["model"].feature_importances_

    # Maps each one-hot column back to its original feature
    rows = []
    for name, imp in zip(all_feature_names, importances):
        if name in numeric_features:
            original = name
        else:
            # one-hot names look like "org_form_code_AS"
            original = name.split("_")[0]
            # Better mapping for our known categorical features
            for cat in categorical_features:
                if name.startswith(cat + "_"):
                    original = cat
                    break
        rows.append({"feature": original, "importance": imp})

    # Sums up the importance of each feature
    imp_df = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=imp_df, x="importance", y="feature", color="steelblue")
    plt.title("Random Forest feature importance")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def main():
    # 1) Loads and cleans
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # 2) Keeps only top 10 industries (no "Other")
    df = filter_top_industries(df, top_n=10)

    # 3) Shows target distribution
    plot_industry_distribution(df)

    # 4) Builds features/target and train models
    X, y, categorical_features, numeric_features = prepare_features(df)
    results_df, rf_pipeline, y_train = train_and_evaluate(
        X, y, categorical_features, numeric_features
    )

    # 5) Visualises results
    plot_model_comparison(results_df)
    plot_feature_importance(rf_pipeline, categorical_features, numeric_features)

    print("\nDone!")


if __name__ == "__main__":
    main()