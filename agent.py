import os
import json
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from tools.clean_data import clean_data
from tools.run_eda import run_eda
from tools.detect_outliers import detect_outliers
from tools.make_charts import make_charts

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Tool definitions for Hugging Face (OpenAI compatible schema) ──────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "clean_data",
            "description": "Clean the uploaded dataset. Removes duplicates, fixes missing values, converts columns to correct types. Always call this first.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_eda",
            "description": "Run exploratory data analysis. Returns shape, column types, numeric statistics, and correlation info.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_outliers",
            "description": "Detect outliers in numeric columns using IQR and Z-score methods.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_charts",
            "description": "Generate visualisation charts: histograms, correlation heatmap, bar charts, boxplots.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": "Write the final plain-English report summarising all findings. Call this last, after all other tools.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# ── Agent class ───────────────────────────────────────────────────────────────

class DataAnalystAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=model_name, token=token)
        self.model_name = model_name
        self.df = None
        self.results = {}

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

    def load_data(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}

    def _execute_tool(self, tool_name: str) -> dict:
        """Execute a tool and store its result."""
        if self.df is None:
            return {"error": "No data loaded"}

        if tool_name == "clean_data":
            self.df, report = clean_data(self.df)
            self.results["clean_data"] = report
            return report

        elif tool_name == "run_eda":
            report = run_eda(self.df)
            self.results["eda"] = report
            return report

        elif tool_name == "detect_outliers":
            report = detect_outliers(self.df)
            self.results["outliers"] = report
            return report

        elif tool_name == "make_charts":
            report = make_charts(self.df)
            self.results["charts"] = report
            return {"chart_count": report["chart_count"], "status": "success"}

        elif tool_name == "write_summary":
            # Model will generate the summary text itself
            return {"status": "ready_to_summarise"}

        return {"error": f"Unknown tool: {tool_name}"}

    def run(self, status_callback=None):
        """
        Run the full agentic loop.
        status_callback(message) is called on each step so the UI can show progress.
        Returns: (final_summary: str, results: dict)
        """
        if self.df is None:
            return "No data loaded.", {}

        # Build the initial prompt
        col_preview = list(self.df.columns)[:10]
        sample_rows = self.df.head(3).to_dict(orient="records")

        prompt = f"""You are an expert data analyst. The user has uploaded a dataset.

Dataset overview:
- Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
- Columns (first 10): {col_preview}
- Sample rows: {json.dumps(sample_rows, default=str)[:800]}

Your job:
1. Call clean_data first to clean the dataset.
2. Call run_eda to understand the data.
3. Call detect_outliers to find anomalies.
4. Call make_charts to generate visualisations.
5. Finally call write_summary to produce a plain-English report.

Use tools in this order. After all tools are done, write a comprehensive summary report.
"""

        messages = [{"role": "user", "content": prompt}]

        max_rounds = 10
        rounds = 0

        while rounds < max_rounds:
            rounds += 1
            
            response = self._call_with_retry(
                self.client.chat_completion,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto"
            )

            choice_message = response.choices[0].message
            
            # Format model response message for appending to conversation history
            assistant_msg = {"role": "assistant"}
            if choice_message.content:
                assistant_msg["content"] = choice_message.content
            if choice_message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments if tc.function.arguments is not None else "{}"
                        }
                    }
                    for tc in choice_message.tool_calls
                ]
            
            messages.append(assistant_msg)

            if not choice_message.tool_calls:
                # No tool calls -> final summary text is ready
                final_summary = choice_message.content or "Analysis complete."
                self.results["summary"] = final_summary.strip()
                return final_summary.strip(), self.results

            # Execute all tool calls in this round
            for tc in choice_message.tool_calls:
                tool_name = tc.function.name
                if status_callback:
                    status_callback(f"Running: {tool_name}...")

                result = self._execute_tool(tool_name)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": json.dumps(result, default=str)[:3000]
                    }
                )

        # Fallback if loop ends without a summary
        return "Analysis complete. See results above.", self.results
