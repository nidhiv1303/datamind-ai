import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def explain_model(model_result: dict, df: pd.DataFrame) -> dict:
    """
    Generates SHAP explanations for the trained model.
    Returns charts and per-feature importance data.
    """
    if not SHAP_AVAILABLE:
        return {"status": "error", "message": "shap not installed. Run: pip install shap"}

    model        = model_result.get("_model")
    scaler       = model_result.get("_scaler")
    le_map       = model_result.get("_le_map", {})
    feature_names = model_result.get("feature_names", [])
    target_col   = model_result.get("target_column", "target")
    task         = model_result.get("task", "classification")

    if model is None:
        return {"status": "error", "message": "No trained model found. Train a model first."}

    charts = []
    result = {"status": "success", "task": task, "target_column": target_col}

    try:
        # ── Prepare data ──────────────────────────────────────────────────────
        X = df[feature_names].copy() if all(f in df.columns for f in feature_names) else None
        if X is None:
            return {"status": "error", "message": "Feature columns not found in dataset."}

        # encode categoricals
        for col in X.select_dtypes(include="object").columns:
            if col in le_map:
                X[col] = le_map[col].transform(X[col].astype(str))
            else:
                X[col] = 0

        X.fillna(X.median(numeric_only=True), inplace=True)

        # use sample for speed (max 200 rows)
        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42).reset_index(drop=True)
        X_scaled = scaler.transform(X_sample)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        # ── SHAP explainer ────────────────────────────────────────────────────
        model_name = model_result.get("best_model", "")

        if "Random Forest" in model_name or "Gradient" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_scaled)
        else:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values_raw = explainer.shap_values(X_scaled)

        # For multiclass classification, use first class or mean
        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 2:
                shap_vals = shap_values_raw[1]  # positive class
            else:
                shap_vals = np.mean(np.abs(shap_values_raw), axis=0)
        else:
            shap_vals = shap_values_raw

        shap_df = pd.DataFrame(shap_vals, columns=feature_names)

        # ── Chart 1: Mean SHAP bar chart ──────────────────────────────────────
        mean_shap = shap_df.abs().mean().sort_values(ascending=False)
        top_features = mean_shap.head(12)

        fig, ax = plt.subplots(figsize=(7, max(3.5, len(top_features) * 0.38)))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))[::-1]
        bars = ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP value| — Average impact on prediction", fontsize=10)
        ax.set_title(f"Feature Importance (SHAP)\nTarget: {target_col}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, top_features.values[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, color="#374151")
        plt.tight_layout()
        charts.append({"title": "Feature Importance (SHAP)", "image_base64": _fig_to_base64(fig)})

        # ── Chart 2: SHAP beeswarm / summary ─────────────────────────────────
        top_cols = mean_shap.head(10).index.tolist()
        shap_top = shap_df[top_cols]
        X_top    = X_scaled_df[top_cols]

        fig, ax = plt.subplots(figsize=(7, max(4, len(top_cols) * 0.5)))
        for i, col in enumerate(top_cols[::-1]):
            vals  = shap_top[col].values
            feats = X_top[col].values
            feat_norm = (feats - feats.min()) / (feats.max() - feats.min() + 1e-9)
            colors_scatter = plt.cm.coolwarm(feat_norm)
            y_jitter = np.random.uniform(-0.2, 0.2, len(vals))
            ax.scatter(vals, np.full(len(vals), i) + y_jitter,
                       c=colors_scatter, alpha=0.6, s=12, linewidths=0)

        ax.set_yticks(range(len(top_cols)))
        ax.set_yticklabels(top_cols[::-1], fontsize=9)
        ax.axvline(0, color="#6b7280", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
        ax.set_title("SHAP Beeswarm Plot\nRed = high feature value, Blue = low", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        charts.append({"title": "SHAP Beeswarm Plot", "image_base64": _fig_to_base64(fig)})

        # ── Chart 3: Single prediction explanation (first row) ────────────────
        row_shap = shap_df.iloc[0]
        row_feat = X_scaled_df.iloc[0]
        top_row  = row_shap.abs().sort_values(ascending=False).head(10)
        row_vals = row_shap[top_row.index]

        fig, ax = plt.subplots(figsize=(7, max(3.5, len(top_row) * 0.42)))
        colors_row = ["#ef4444" if v > 0 else "#3b82f6" for v in row_vals.values[::-1]]
        ax.barh(top_row.index[::-1], row_vals.values[::-1], color=colors_row)
        ax.axvline(0, color="#1a1730", linewidth=0.8)
        ax.set_xlabel("SHAP value", fontsize=10)
        ax.set_title("Single Prediction Explanation (Sample Row 1)\nRed = pushes prediction up · Blue = pushes down",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        charts.append({"title": "Single Prediction Explanation", "image_base64": _fig_to_base64(fig)})

        # ── Top feature insights ──────────────────────────────────────────────
        top5 = []
        for feat in mean_shap.head(5).index:
            shap_col = shap_df[feat]
            feat_col = X_scaled_df[feat]
            corr = float(np.corrcoef(feat_col, shap_col)[0, 1])
            direction = "increases" if corr > 0 else "decreases"
            top5.append({
                "feature": feat,
                "mean_impact": round(float(mean_shap[feat]), 4),
                "direction": direction,
                "insight": f"Higher {feat} {direction} the predicted {target_col}"
            })

        result["charts"]       = charts
        result["top_features"] = top5
        result["feature_count"] = len(feature_names)
        result["sample_size"]  = sample_size

    except Exception as e:
        return {"status": "error", "message": f"SHAP error: {str(e)}"}

    return result
