import os
import json
import pandas as pd
from huggingface_hub import InferenceClient
import dotenv

dotenv.load_dotenv()
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


class ChatAgent:
    """
    A conversational agent that answers questions about the analysed dataset.
    It uses the EDA results, outlier report, and a data sample as context.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=model_name, token=token)
        self.model_name = model_name
        self.history = []
        self.context = ""

    def _call_with_retry(self, func, *args, **kwargs):
        import time
        import random
        max_retries = 5
        delay = 2.0
        backoff_factor = 2.0
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str or 
                    "rate limit" in err_str or 
                    "too many requests" in err_str or
                    "503" in err_str or
                    "unavailable" in err_str
                )
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = delay + random.uniform(0, 1.0)
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                else:
                    raise e

    def load_context(self, df: pd.DataFrame, results: dict, summary: str):
        """
        Build the context string from analysis results.
        Called once after the Analyst Agent finishes.
        """
        eda = results.get("eda", {})
        outliers = results.get("outliers", {})
        clean = results.get("clean_data", {})

        # Columns list truncation
        col_names = eda.get('column_names', [])
        if len(col_names) > 30:
            col_names_str = ', '.join(col_names[:25]) + f" ... and {len(col_names) - 25} more"
        else:
            col_names_str = ', '.join(col_names)

        # Data sample (first 5 rows as markdown/string table) - limit columns
        cols_to_show = list(df.columns)
        if len(cols_to_show) > 15:
            cols_to_show = cols_to_show[:12]
            df_sample = df[cols_to_show].head(5)
            sample_suffix = f"\n\n* (Showing first 12 columns out of {df.shape[1]})"
        else:
            df_sample = df.head(5)
            sample_suffix = ""

        try:
            sample_md = df_sample.to_markdown(index=False) + sample_suffix
        except Exception:
            sample_md = df_sample.to_string(index=False) + sample_suffix

        # Numeric stats - limit columns
        numeric_stats = ""
        if "numeric_stats" in eda:
            stats_df = pd.DataFrame(eda["numeric_stats"])
            if stats_df.shape[1] > 15:
                stats_df = stats_df.iloc[:, :12]
                stats_suffix = f"\n\n* (Showing first 12 numeric columns out of {pd.DataFrame(eda['numeric_stats']).shape[1]})"
            else:
                stats_suffix = ""
            try:
                numeric_stats = stats_df.round(2).to_markdown() + stats_suffix
            except Exception:
                numeric_stats = stats_df.to_string() + stats_suffix

        # Correlations - limit items
        corr_text = ""
        if eda.get("high_correlations"):
            corrs = eda["high_correlations"]
            if len(corrs) > 20:
                corr_text = "\n".join([
                    f"  - {c['col1']} ↔ {c['col2']}: {c['correlation']}"
                    for c in corrs[:20]
                ]) + f"\n  - ... and {len(corrs) - 20} more correlations."
            else:
                corr_text = "\n".join([
                    f"  - {c['col1']} ↔ {c['col2']}: {c['correlation']}"
                    for c in corrs
                ])

        # Outliers - limit items
        outlier_text = ""
        if outliers.get("columns_with_outliers"):
            details = outliers.get("outlier_details", {})
            col_outliers = list(details.items())
            if len(col_outliers) > 20:
                outlier_text = "\n".join([
                    f"  - {col}: {info['iqr_outlier_count']} IQR outliers, range {info['min_value']}→{info['max_value']}"
                    for col, info in col_outliers[:20]
                ]) + f"\n  - ... and {len(col_outliers) - 20} more columns with outliers."
            else:
                outlier_text = "\n".join([
                    f"  - {col}: {info['iqr_outlier_count']} IQR outliers, range {info['min_value']}→{info['max_value']}"
                    for col, info in col_outliers
                ])

        # Categorical info - limit items
        cat_text = ""
        if "categorical_info" in eda:
            cat_cols = list(eda["categorical_info"].items())
            lines = []
            if len(cat_cols) > 20:
                for col, info in cat_cols[:20]:
                    top = list(info["top_3"].keys())[:3]
                    lines.append(f"  - {col}: {info['unique_values']} unique values, top: {top}")
                lines.append(f"  - ... and {len(cat_cols) - 20} more categorical columns.")
            else:
                for col, info in cat_cols:
                    top = list(info["top_3"].keys())[:3]
                    lines.append(f"  - {col}: {info['unique_values']} unique values, top: {top}")
            cat_text = "\n".join(lines)

        self.context = f"""
You are a helpful data analyst assistant. The user has uploaded a dataset and it has already been analysed.
Answer all questions based ONLY on the data context below. Be concise, specific, and use numbers where possible.
If you cannot answer from the context, say so honestly.

=== DATASET INFO ===
Shape: {eda.get('rows', '?')} rows × {eda.get('columns', '?')} columns
Columns: {col_names_str}
Duplicates removed: {clean.get('duplicates_removed', 0)}
Missing values filled: {clean.get('missing_values_filled', 0)}

=== SAMPLE DATA (first 5 rows) ===
{sample_md}

=== NUMERIC STATISTICS ===
{numeric_stats}

=== HIGH CORRELATIONS (|r| > 0.5) ===
{corr_text if corr_text else 'None found'}

=== OUTLIERS ===
{outlier_text if outlier_text else 'No significant outliers found'}

=== CATEGORICAL COLUMNS ===
{cat_text if cat_text else 'None'}

=== AI ANALYSIS SUMMARY ===
{summary}
""".strip()

        # Initialize conversation history with system instruction context
        self.history = [
            {"role": "system", "content": self.context}
        ]

    def ask(self, question: str) -> str:
        """Send a question and get an answer."""
        if not self.history:
            return "Please run the analysis first before asking questions."
        try:
            self.history.append({"role": "user", "content": question})
            response = self._call_with_retry(
                self.client.chat_completion,
                messages=self.history
            )
            ans = response.choices[0].message.content or ""
            self.history.append({"role": "assistant", "content": ans})
            return ans.strip()
        except Exception as e:
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()  # Clean up failed prompt
            return f"Error: {e}"

    def reset(self):
        """Reset the chat history but keep the context."""
        if self.context:
            self.history = [
                {"role": "system", "content": self.context}
            ]
        else:
            self.history = []
        return "Chat history cleared."
