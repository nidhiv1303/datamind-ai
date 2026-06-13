import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import json
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _detect_task(series: pd.Series) -> str:
    """Detect if target is classification or regression."""
    if series.dtype == object:
        return "classification"
    unique_ratio = series.nunique() / len(series)
    if series.nunique() <= 10 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def _preprocess(df: pd.DataFrame, target_col: str):
    """Prepare features and target for training."""
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop columns with too many unique strings (like IDs, names)
    for col in X.select_dtypes(include="object").columns:
        if X[col].nunique() > 50:
            X.drop(columns=[col], inplace=True)

    # Encode categorical features
    le_map = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    # Fill any remaining nulls
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Encode target if classification
    target_le = None
    if y.dtype == object or y.nunique() <= 10:
        target_le = LabelEncoder()
        y = target_le.fit_transform(y.astype(str))

    feature_names = list(X.columns)
    return X.values, y, feature_names, target_le, le_map


def train_model(df: pd.DataFrame, target_col: str) -> dict:
    """
    Auto-detects task type, trains best model, returns metrics + charts.
    """
    if target_col not in df.columns:
        return {"status": "error", "message": f"Column '{target_col}' not found."}

    if len(df) < 20:
        return {"status": "error", "message": "Need at least 20 rows to train a model."}

    task = _detect_task(df[target_col])
    result = {"task": task, "target_column": target_col}

    try:
        X, y, feature_names, target_le, le_map = _preprocess(df, target_col)
    except Exception as e:
        return {"status": "error", "message": f"Preprocessing error: {e}"}

    # Train/test split
    test_size = 0.2 if len(df) > 100 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train multiple models, pick best ──────────────────────────────────────
    charts = []

    if task == "classification":
        models = {
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        }
        scores = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            scores[name] = accuracy_score(y_test, m.predict(X_test))

        best_name = max(scores, key=scores.get)
        best_model = models[best_name]
        y_pred = best_model.predict(X_test)

        result["best_model"]  = best_name
        result["accuracy"]    = round(scores[best_name] * 100, 2)
        result["all_scores"]  = {k: round(v * 100, 2) for k, v in scores.items()}

        # Classification report
        labels = target_le.classes_ if target_le else None
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        result["classification_report"] = report

        # Confusion matrix chart
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(f"Confusion Matrix — {best_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        charts.append({"title": "Confusion Matrix", "image_base64": _fig_to_base64(fig)})

    else:  # regression
        models = {
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
        }
        scores = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            scores[name] = r2_score(y_test, m.predict(X_test))

        best_name  = max(scores, key=scores.get)
        best_model = models[best_name]
        y_pred     = best_model.predict(X_test)

        result["best_model"] = best_name
        result["r2_score"]   = round(scores[best_name], 4)
        result["mae"]        = round(mean_absolute_error(y_test, y_pred), 4)
        result["rmse"]       = round(mean_squared_error(y_test, y_pred) ** 0.5, 4)
        result["all_scores"] = {k: round(v, 4) for k, v in scores.items()}

        # Actual vs Predicted chart
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.5, color="#6366f1", edgecolors="white", linewidth=0.5)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted — {best_name}")
        ax.legend()
        charts.append({"title": "Actual vs Predicted", "image_base64": _fig_to_base64(fig)})

    # ── Feature importance (works for RF and GB) ──────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(6, max(3, len(fi_df) * 0.35)))
        sns.barplot(data=fi_df, x="importance", y="feature", ax=ax, palette="viridis")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        charts.append({"title": "Feature Importance", "image_base64": _fig_to_base64(fig)})

        result["top_features"] = fi_df.head(5).to_dict(orient="records")

    # ── Model comparison chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    metric_label = "Accuracy (%)" if task == "classification" else "R² Score"
    model_names  = list(result["all_scores"].keys())
    model_scores = list(result["all_scores"].values())
    colors = ["#6366f1" if n == best_name else "#d1d5db" for n in model_names]
    bars = ax.barh(model_names, model_scores, color=colors)
    ax.set_xlabel(metric_label)
    ax.set_title("Model Comparison")
    for bar, score in zip(bars, model_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                str(score), va="center", fontsize=10)
    charts.append({"title": "Model Comparison", "image_base64": _fig_to_base64(fig)})

    # ── Store model artifacts for prediction ──────────────────────────────────
    result["charts"]        = charts
    result["feature_names"] = feature_names
    result["status"]        = "success"
    result["train_size"]    = len(X_train)
    result["test_size"]     = len(X_test)

    # Store model + preprocessors in result for live prediction
    result["_model"]     = best_model
    result["_scaler"]    = scaler
    result["_le_map"]    = le_map
    result["_target_le"] = target_le

    return result


def predict_single(model_result: dict, input_values: dict) -> str:
    """
    Make a single prediction given a dict of feature_name → value.
    """
    try:
        model     = model_result["_model"]
        scaler    = model_result["_scaler"]
        le_map    = model_result["_le_map"]
        target_le = model_result["_target_le"]
        features  = model_result["feature_names"]

        row = []
        for f in features:
            val = input_values.get(f, 0)
            if f in le_map:
                try:
                    val = le_map[f].transform([str(val)])[0]
                except Exception:
                    val = 0
            row.append(float(val))

        X = scaler.transform([row])
        pred = model.predict(X)[0]

        if target_le is not None:
            pred = target_le.inverse_transform([int(pred)])[0]

        return str(pred)
    except Exception as e:
        return f"Prediction error: {e}"
