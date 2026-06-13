import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def make_charts(df: pd.DataFrame) -> dict:
    """
    Generates charts for numeric and categorical columns.
    Returns base64-encoded PNG images.
    """
    charts = []
    numeric_df = df.select_dtypes(include=["number"])
    cat_df = df.select_dtypes(include=["object"])

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Distribution plots for numeric columns (max 4)
    for col in list(numeric_df.columns)[:4]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#6366f1")
        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(col)
        charts.append({
            "title": f"Distribution of {col}",
            "type": "histogram",
            "image_base64": _fig_to_base64(fig)
        })

    # 2. Correlation heatmap (if 2+ numeric columns)
    if len(numeric_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        corr = numeric_df.corr().round(2)
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f",
                    linewidths=0.5, square=True)
        ax.set_title("Correlation Heatmap", fontsize=12)
        charts.append({
            "title": "Correlation Heatmap",
            "type": "heatmap",
            "image_base64": _fig_to_base64(fig)
        })

    # 3. Bar chart for top categorical column (if any)
    if not cat_df.empty:
        col = cat_df.columns[0]
        top_vals = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax, palette="muted")
        ax.set_title(f"Top values in '{col}'", fontsize=12)
        ax.set_xlabel("Count")
        charts.append({
            "title": f"Top values in '{col}'",
            "type": "bar",
            "image_base64": _fig_to_base64(fig)
        })

    # 4. Boxplot for numeric columns (outlier view)
    if not numeric_df.empty:
        cols_to_plot = list(numeric_df.columns)[:5]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df[cols_to_plot].boxplot(ax=ax)
        ax.set_title("Boxplot — outlier overview", fontsize=12)
        plt.xticks(rotation=30, ha="right")
        charts.append({
            "title": "Boxplot — outlier overview",
            "type": "boxplot",
            "image_base64": _fig_to_base64(fig)
        })

    return {"charts": charts, "chart_count": len(charts), "status": "success"}
