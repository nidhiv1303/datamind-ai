import pandas as pd
import json


def clean_data(df: pd.DataFrame) -> dict:
    """
    Cleans the dataframe: handles missing values, fixes dtypes,
    removes duplicates. Returns a summary of what was done.
    """
    report = {}
    original_shape = df.shape

    # 1. Remove fully empty rows/columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # 2. Remove duplicate rows
    dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = int(dupes)

    # 3. Fix numeric columns stored as strings
    fixed_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col].str.replace(",", "").str.strip(), errors="coerce")
            if converted.notna().sum() > len(df) * 0.6:
                df[col] = converted
                fixed_cols.append(col)
    report["columns_converted_to_numeric"] = fixed_cols

    # 4. Fill missing values
    missing_before = df.isnull().sum().sum()
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    report["missing_values_filled"] = int(missing_before)

    # 5. Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    report["original_shape"] = list(original_shape)
    report["cleaned_shape"] = list(df.shape)
    report["status"] = "success"

    return df, report
