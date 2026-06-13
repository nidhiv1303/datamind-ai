import pandas as pd
import numpy as np
from scipy import stats


def score_data_quality(df: pd.DataFrame) -> dict:
    """
    Scores dataset quality out of 100 across 6 categories.
    Returns scores, grades, issues, and overall report card.
    """
    results = {}
    total_cells = df.shape[0] * df.shape[1]

    # ── 1. Completeness (missing values) ─────────────────────────────────────
    missing = df.isnull().sum().sum()
    missing_pct = (missing / total_cells) * 100
    completeness_score = max(0, 100 - (missing_pct * 5))
    completeness_score = min(100, round(completeness_score, 1))

    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0].to_dict()

    results["completeness"] = {
        "score": completeness_score,
        "grade": _grade(completeness_score),
        "missing_pct": round(missing_pct, 2),
        "missing_cells": int(missing),
        "affected_columns": {k: int(v) for k, v in missing_cols.items()},
        "insight": f"{missing_pct:.1f}% of cells are missing" if missing_pct > 0 else "No missing values — perfect!"
    }

    # ── 2. Uniqueness (duplicates) ────────────────────────────────────────────
    dupes = df.duplicated().sum()
    dupe_pct = (dupes / len(df)) * 100
    uniqueness_score = max(0, 100 - (dupe_pct * 10))
    uniqueness_score = min(100, round(uniqueness_score, 1))

    results["uniqueness"] = {
        "score": uniqueness_score,
        "grade": _grade(uniqueness_score),
        "duplicate_rows": int(dupes),
        "duplicate_pct": round(dupe_pct, 2),
        "insight": f"{dupes} duplicate rows found ({dupe_pct:.1f}%)" if dupes > 0 else "No duplicate rows found!"
    }

    # ── 3. Consistency (data types & formatting) ──────────────────────────────
    issues = []
    for col in df.select_dtypes(include="object").columns:
        # check if numeric data stored as string
        converted = pd.to_numeric(df[col].str.replace(",", "").str.strip(), errors="coerce")
        if converted.notna().sum() > len(df) * 0.7:
            issues.append(f"'{col}' looks numeric but stored as text")
        # check mixed types
        if df[col].apply(type).nunique() > 1:
            issues.append(f"'{col}' has mixed data types")

    # check for whitespace issues
    for col in df.select_dtypes(include="object").columns:
        if df[col].str.contains(r"^\s|\s$", regex=True, na=False).any():
            issues.append(f"'{col}' has leading/trailing whitespace")

    consistency_score = max(0, 100 - (len(issues) * 15))
    consistency_score = min(100, round(consistency_score, 1))

    results["consistency"] = {
        "score": consistency_score,
        "grade": _grade(consistency_score),
        "issues_found": issues,
        "issue_count": len(issues),
        "insight": f"{len(issues)} consistency issues found" if issues else "Data types and formats look consistent!"
    }

    # ── 4. Validity (outliers & range) ────────────────────────────────────────
    numeric_df = df.select_dtypes(include="number")
    outlier_cols = []
    total_outliers = 0

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            outlier_cols.append({"column": col, "count": int(outliers)})
            total_outliers += outliers

    outlier_pct = (total_outliers / max(1, len(numeric_df) * len(numeric_df.columns))) * 100
    validity_score = max(0, 100 - (outlier_pct * 3))
    validity_score = min(100, round(validity_score, 1))

    results["validity"] = {
        "score": validity_score,
        "grade": _grade(validity_score),
        "outlier_columns": outlier_cols,
        "total_outliers": int(total_outliers),
        "insight": f"{total_outliers} outliers across {len(outlier_cols)} columns" if total_outliers > 0 else "No significant outliers found!"
    }

    # ── 5. Distribution (skewness) ────────────────────────────────────────────
    skewed_cols = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue
        skew = abs(series.skew())
        if skew > 1:
            skewed_cols.append({"column": col, "skewness": round(float(skew), 2)})

    skew_pct = (len(skewed_cols) / max(1, len(numeric_df.columns))) * 100
    distribution_score = max(0, 100 - (skew_pct * 0.8))
    distribution_score = min(100, round(distribution_score, 1))

    results["distribution"] = {
        "score": distribution_score,
        "grade": _grade(distribution_score),
        "skewed_columns": skewed_cols,
        "insight": f"{len(skewed_cols)} columns are highly skewed" if skewed_cols else "Distributions look balanced!"
    }

    # ── 6. Balance (class imbalance for categorical cols) ─────────────────────
    cat_df = df.select_dtypes(include="object")
    imbalanced = []
    for col in cat_df.columns:
        vc = df[col].value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] > 0.85:
            imbalanced.append({"column": col, "dominant_pct": round(float(vc.iloc[0]) * 100, 1)})

    balance_score = max(0, 100 - (len(imbalanced) * 20))
    balance_score = min(100, round(balance_score, 1))

    results["balance"] = {
        "score": balance_score,
        "grade": _grade(balance_score),
        "imbalanced_columns": imbalanced,
        "insight": f"{len(imbalanced)} columns have class imbalance" if imbalanced else "Class distribution looks balanced!"
    }

    # ── Overall Score ─────────────────────────────────────────────────────────
    weights = {
        "completeness": 0.25,
        "uniqueness": 0.15,
        "consistency": 0.20,
        "validity": 0.20,
        "distribution": 0.10,
        "balance": 0.10,
    }
    overall = sum(results[k]["score"] * w for k, w in weights.items())
    overall = round(overall, 1)

    # Recommendations
    recommendations = []
    if results["completeness"]["score"] < 80:
        recommendations.append("Handle missing values before modelling — consider imputation or dropping columns with >30% missing.")
    if results["uniqueness"]["score"] < 90:
        recommendations.append("Remove duplicate rows to avoid model bias.")
    if results["consistency"]["score"] < 80:
        recommendations.append("Fix data type inconsistencies — convert numeric-looking string columns.")
    if results["validity"]["score"] < 80:
        recommendations.append("Treat outliers — consider capping, removing, or transforming extreme values.")
    if results["distribution"]["score"] < 80:
        recommendations.append("Apply log or Box-Cox transformation to reduce skewness in numeric columns.")
    if results["balance"]["score"] < 80:
        recommendations.append("Address class imbalance — consider SMOTE or class weights during modelling.")
    if not recommendations:
        recommendations.append("Dataset looks clean and ready for modelling!")

    return {
        "categories": results,
        "overall_score": overall,
        "overall_grade": _grade(overall),
        "recommendations": recommendations,
        "status": "success"
    }


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"
