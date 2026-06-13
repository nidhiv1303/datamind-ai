import pandas as pd
import json


def run_eda(df: pd.DataFrame) -> dict:
    """
    Runs exploratory data analysis on the dataframe.
    Returns statistics, column info, correlations.
    """
    report = {}

    # Basic info
    report["rows"] = len(df)
    report["columns"] = len(df.columns)
    report["column_names"] = list(df.columns)

    # Column types
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Numeric stats
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        stats = numeric_df.describe().round(2)
        report["numeric_stats"] = stats.to_dict()

        # Correlation matrix (top pairs only)
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr().round(2)
            pairs = []
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.5:
                        pairs.append({
                            "col1": cols[i],
                            "col2": cols[j],
                            "correlation": float(val)
                        })
            pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            report["high_correlations"] = pairs[:5]

    # Categorical stats
    cat_df = df.select_dtypes(include=["object"])
    if not cat_df.empty:
        cat_info = {}
        for col in cat_df.columns:
            vc = df[col].value_counts()
            cat_info[col] = {
                "unique_values": int(df[col].nunique()),
                "top_3": vc.head(3).to_dict()
            }
        report["categorical_info"] = cat_info

    report["status"] = "success"
    return report
