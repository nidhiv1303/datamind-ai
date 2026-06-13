import pandas as pd
import numpy as np
from scipy import stats


def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detects outliers in numeric columns using IQR and Z-score methods.
    """
    report = {}
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return {"status": "no_numeric_columns", "outliers": {}}

    outlier_info = {}

    for col in numeric_df.columns:
        series = df[col].dropna()

        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        iqr_outliers = series[(series < lower) | (series > upper)]

        # Z-score method
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3]

        if len(iqr_outliers) > 0 or len(z_outliers) > 0:
            outlier_info[col] = {
                "iqr_outlier_count": int(len(iqr_outliers)),
                "iqr_lower_bound": round(float(lower), 2),
                "iqr_upper_bound": round(float(upper), 2),
                "zscore_outlier_count": int(len(z_outliers)),
                "min_value": round(float(series.min()), 2),
                "max_value": round(float(series.max()), 2),
                "sample_outlier_values": [round(float(v), 2) for v in iqr_outliers.head(5).tolist()]
            }

    report["columns_with_outliers"] = list(outlier_info.keys())
    report["outlier_details"] = outlier_info
    report["total_columns_checked"] = len(numeric_df.columns)
    report["status"] = "success"

    return report
