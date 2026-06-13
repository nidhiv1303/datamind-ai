import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime
from jinja2 import Template
import os

from agent import DataAnalystAgent
from chat_agent import ChatAgent
from tools.automl import train_model, predict_single
from tools.data_quality import score_data_quality
from tools.explainability import explain_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  .stApp { background: #f8f7ff; }

  [data-testid="stSidebar"] {
    background: #1a1730 !important;
    border-right: 1px solid rgba(99,102,241,0.15);
  }
  [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #fff !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
  [data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #fff !important; border-radius: 8px !important;
  }
  [data-testid="stSidebar"] .stButton > button {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important; border-radius: 8px !important;
    width: 100%;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.3) !important;
  }

  .main-header {
    background: #6366f1;
    color: white; padding: 28px 32px;
    border-radius: 14px; margin-bottom: 28px;
    border-bottom: 3px solid #4f46e5;
  }
  .main-header h1 { font-size: 24px; font-weight: 600; letter-spacing: -0.3px; }
  .main-header p  { opacity: 0.8; margin-top: 6px; font-size: 13px; }

  [data-testid="stMetric"] {
    background: white; border: 1px solid #e5e3f7;
    border-radius: 12px; padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(99,102,241,0.06);
  }
  [data-testid="stMetricLabel"] { font-size: 12px !important; color: #7c7aad !important; font-weight: 500 !important; }
  [data-testid="stMetricValue"] { font-size: 28px !important; color: #4f46e5 !important; font-weight: 600 !important; }

  [data-testid="stDataFrame"] { border: 1px solid #e5e3f7 !important; border-radius: 10px !important; overflow: hidden; }

  .stTabs [data-baseweb="tab-list"] { background: #eeedf8; border-radius: 10px; padding: 4px; gap: 2px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-size: 13px !important; padding: 6px 14px !important; color: #6b6a9e !important; font-weight: 500 !important; }
  .stTabs [aria-selected="true"] { background: white !important; color: #4f46e5 !important; box-shadow: 0 1px 3px rgba(99,102,241,0.12) !important; }

  .stButton > button[kind="primary"] {
    background: #6366f1 !important; border: none !important;
    border-radius: 10px !important; font-weight: 500 !important;
    font-size: 15px !important; padding: 12px 0 !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
  }
  .stButton > button[kind="primary"]:hover { background: #4f46e5 !important; }

  [data-testid="stExpander"] { border: 1px solid #e5e3f7 !important; border-radius: 10px !important; background: white !important; }
  [data-testid="stFileUploader"] { border: 1.5px dashed #c4c2f0 !important; border-radius: 12px !important; background: white !important; }
  hr { border-color: #e5e3f7 !important; }
  [data-testid="stAlert"] { border-radius: 10px !important; }

  .agent-step {
    background: rgba(99,102,241,0.12); border-left: 3px solid #6366f1;
    padding: 8px 14px; border-radius: 0 8px 8px 0;
    font-size: 13px; margin: 4px 0; color: #a5b4fc !important;
  }

  /* Quality scorecard */
  .scorecard-ring {
    width: 120px; height: 120px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; margin: 0 auto 16px;
    font-size: 32px; font-weight: 700; color: white;
  }
  .scorecard-ring span { font-size: 13px; font-weight: 400; opacity: 0.85; }
  .category-card {
    background: white; border: 1px solid #e5e3f7;
    border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 3px rgba(99,102,241,0.05);
  }
  .category-card .cat-name { font-size: 13px; font-weight: 600; color: #1a1730; margin-bottom: 4px; }
  .category-card .cat-score { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
  .category-card .cat-insight { font-size: 11px; color: #7c7aad; line-height: 1.5; }
  .grade-badge {
    display: inline-block; width: 28px; height: 28px; border-radius: 6px;
    text-align: center; line-height: 28px; font-size: 13px;
    font-weight: 700; float: right; margin-top: -2px;
  }
  .progress-bar-bg { background: #eeedf8; border-radius: 4px; height: 6px; margin: 8px 0; }
  .progress-bar-fill { height: 6px; border-radius: 4px; transition: width 0.3s; }

  /* What-if simulator */
  .whatif-result {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; border-radius: 14px; padding: 24px;
    text-align: center; margin-top: 16px;
  }
  .whatif-result .prediction-label { font-size: 13px; opacity: 0.8; margin-bottom: 6px; }
  .whatif-result .prediction-value { font-size: 36px; font-weight: 700; }
  .whatif-result .prediction-note { font-size: 11px; opacity: 0.7; margin-top: 6px; }

  /* metric card custom */
  .metric-card { background: white; border: 1px solid #e5e3f7; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(99,102,241,0.06); }
  .metric-card .value { font-size: 30px; font-weight: 600; color: #4f46e5; }
  .metric-card .label { font-size: 12px; color: #7c7aad; margin-top: 4px; font-weight: 500; }

  /* chat */
  .chat-container { border: 1px solid #e5e3f7; border-radius: 14px; padding: 20px; background: white; margin-top: 8px; }
  .chat-msg-user { background: #6366f1; color: white; padding: 10px 16px; border-radius: 18px 18px 4px 18px; margin: 8px 0 8px 20%; font-size: 14px; line-height: 1.6; }
  .chat-msg-ai { background: #f4f3fe; border: 1px solid #e5e3f7; color: #1a1730; padding: 10px 16px; border-radius: 18px 18px 18px 4px; margin: 8px 20% 8px 0; font-size: 14px; line-height: 1.6; }
  .chat-label { font-size: 11px; color: #a0a0c0; margin: 2px 4px; }

  .predict-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px; margin-top: 16px; }
  .predict-result { font-size: 28px; font-weight: 600; color: #059669; text-align: center; padding: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "analysis_done": False, "results": {}, "summary": "",
    "df_clean": None, "chat_agent": None, "chat_history": [],
    "model_result": None, "quality_result": None,
    "shap_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>Intelligent Data Analysis &amp; AutoML Platform</h1>
  <p>Multi-Agent AI System &nbsp;·&nbsp; Analyse · Score · Train · Simulate · Chat</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Hugging Face Token", type="password", placeholder="hf_xxxxxxxx")
    if api_key:
        os.environ["HF_API_KEY"] = api_key
        st.markdown("<p style='color:#4ade80;font-size:12px;'>● Connected</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Agent Architecture")
    st.markdown("""
**Agent 1 — Analyst**
Cleans → EDA → Outliers → Charts → Report

**Agent 2 — Chat (RAG)**
Answers questions using analysis as context

**AutoML Engine**
Trains + evaluates ML models automatically
    """)

    st.markdown("---")
    steps = ["1. Clean data", "2. Run EDA", "3. Detect outliers", "4. Generate charts", "5. Write AI summary"]
    for s in steps:
        st.markdown(f"<div class='agent-step'>{s}</div>", unsafe_allow_html=True)

    if st.session_state.analysis_done:
        st.markdown("---")
        if st.button("🔄 Reset everything"):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file to get started.")
    st.stop()

@st.cache_data
def load_file(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

try:
    df_raw = load_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.markdown("### 👀 Data Preview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df_raw.shape[0])
c2.metric("Columns", df_raw.shape[1])
c3.metric("Numeric cols", len(df_raw.select_dtypes(include="number").columns))
c4.metric("Text cols", len(df_raw.select_dtypes(include="object").columns))
st.dataframe(df_raw.head(10), use_container_width=True)
st.markdown("---")

# ── Run Analysis ──────────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    if not os.getenv("HF_API_KEY"):
        st.warning("⚠️ Enter your Hugging Face token in the sidebar.")
        st.stop()

    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Agent 1 (Analyst) working..."):
            steps_done = []
            status_box = st.empty()

            def update_status(msg):
                steps_done.append(msg)
                status_box.markdown(
                    "\n".join([f"<div class='agent-step'>✅ {s}</div>" for s in steps_done]),
                    unsafe_allow_html=True,
                )

            agent = DataAnalystAgent()
            agent.load_data(df_raw)
            try:
                summary, results = agent.run(status_callback=update_status)
            except Exception as e:
                st.error(f"Analyst agent error: {e}")
                st.stop()

            st.session_state.results  = results
            st.session_state.summary  = summary
            st.session_state.df_clean = agent.df

        with st.spinner("Agent 2 (Chat) loading context..."):
            chat_agent = ChatAgent()
            chat_agent.load_context(agent.df, results, summary)
            st.session_state.chat_agent = chat_agent

        with st.spinner("Scoring data quality..."):
            st.session_state.quality_result = score_data_quality(agent.df)

        st.session_state.analysis_done = True
        st.success("✅ All agents ready!")
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    results  = st.session_state.results
    summary  = st.session_state.summary
    df_clean = st.session_state.df_clean
    qr       = st.session_state.quality_result

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏆 Quality Score",
        "🎯 What-If Simulator",
        "🤖 AutoML",
        "🔍 XAI Explainer",
        "💬 Chat",
        "📋 Summary",
        "📈 Charts",
        "📥 Report",
    ])

    # ── TAB 1: QUALITY SCORECARD ───────────────────────────────────────────────
    with tab1:
        st.markdown("#### 🏆 Dataset Quality Report Card")
        st.caption("Scores your dataset across 6 dimensions before modelling — like a health check for your data.")

        if qr:
            overall = qr["overall_score"]
            grade   = qr["overall_grade"]

            # Color based on score
            ring_color = (
                "linear-gradient(135deg,#22c55e,#16a34a)" if overall >= 75 else
                "linear-gradient(135deg,#f59e0b,#d97706)" if overall >= 50 else
                "linear-gradient(135deg,#ef4444,#dc2626)"
            )

            col_ring, col_recs = st.columns([1, 2])

            with col_ring:
                st.markdown(f"""
                <div style='text-align:center;padding:20px;'>
                  <div class='scorecard-ring' style='background:{ring_color};box-shadow:0 4px 20px rgba(0,0,0,0.15);'>
                    <div>{overall}</div>
                    <span>/ 100</span>
                  </div>
                  <div style='font-size:14px;color:#1a1730;font-weight:600;'>Overall Grade: <span style='color:#4f46e5;font-size:20px;'>{grade}</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col_recs:
                st.markdown("**💡 Recommendations**")
                for rec in qr["recommendations"]:
                    st.markdown(f"- {rec}")

            st.markdown("---")
            st.markdown("#### Category Breakdown")

            cat_labels = {
                "completeness":  ("📋 Completeness",  "Missing values"),
                "uniqueness":    ("🔁 Uniqueness",    "Duplicate rows"),
                "consistency":   ("✅ Consistency",   "Type & format issues"),
                "validity":      ("📏 Validity",      "Outliers"),
                "distribution":  ("📊 Distribution",  "Skewness"),
                "balance":       ("⚖️ Balance",       "Class imbalance"),
            }

            grade_colors = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}

            cols = st.columns(3)
            for i, (key, (label, sublabel)) in enumerate(cat_labels.items()):
                cat = qr["categories"][key]
                score = cat["score"]
                g = cat["grade"]
                gcolor = grade_colors.get(g, "#6366f1")
                bar_color = gcolor

                fill_pct = score

                with cols[i % 3]:
                    st.markdown(f"""
                    <div class='category-card'>
                      <div class='cat-name'>{label}
                        <span class='grade-badge' style='background:{gcolor}20;color:{gcolor};'>{g}</span>
                      </div>
                      <div class='cat-score' style='color:{gcolor};'>{score}</div>
                      <div class='progress-bar-bg'>
                        <div class='progress-bar-fill' style='width:{fill_pct}%;background:{bar_color};'></div>
                      </div>
                      <div class='cat-insight'>{cat['insight']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")

    # ── TAB 2: WHAT-IF SIMULATOR ───────────────────────────────────────────────
    with tab2:
        st.markdown("#### 🎯 What-If Scenario Simulator")
        st.caption("Train a model first (AutoML tab), then use sliders to simulate scenarios and see predictions update instantly.")

        if st.session_state.model_result is None:
            st.info("👉 Go to the **AutoML** tab first, select a target column and train a model. Then come back here.")
        else:
            mr = st.session_state.model_result
            st.success(f"Model ready: **{mr['best_model']}** predicting **{mr['target_column']}**")

            st.markdown("---")
            st.markdown("#### Adjust feature values and see the prediction change")

            feature_names = mr["feature_names"]
            input_vals = {}

            # Two columns layout for sliders
            left_features = feature_names[:len(feature_names)//2 + len(feature_names)%2]
            right_features = feature_names[len(feature_names)//2 + len(feature_names)%2:]

            col_l, col_r = st.columns(2)

            def render_input(feat, container):
                with container:
                    if feat in mr.get("_le_map", {}):
                        le = mr["_le_map"][feat]
                        options = list(le.classes_)
                        return st.selectbox(f"**{feat}**", options, key=f"wi_{feat}")
                    else:
                        if feat in df_clean.columns:
                            mn = float(df_clean[feat].min())
                            mx = float(df_clean[feat].max())
                            med = float(df_clean[feat].median())
                            step = round((mx - mn) / 100, 4) if mx != mn else 1.0
                            return st.slider(f"**{feat}**", min_value=mn, max_value=mx, value=med, step=step, key=f"wi_{feat}")
                        else:
                            return st.number_input(f"**{feat}**", value=0.0, key=f"wi_{feat}")

            for feat in left_features:
                input_vals[feat] = render_input(feat, col_l)
            for feat in right_features:
                input_vals[feat] = render_input(feat, col_r)

            # Live prediction — updates automatically as sliders move
            prediction = predict_single(mr, input_vals)

            task = mr.get("task", "classification")
            if task == "classification":
                pred_display = str(prediction)
                note = f"Predicted class · Model: {mr['best_model']} · Accuracy: {mr.get('accuracy', '?')}%"
            else:
                try:
                    pred_display = f"{float(prediction):,.2f}"
                except Exception:
                    pred_display = str(prediction)
                note = f"Predicted value · Model: {mr['best_model']} · R²: {mr.get('r2_score', '?')}"

            st.markdown(f"""
            <div class='whatif-result'>
              <div class='prediction-label'>Predicted {mr['target_column']}</div>
              <div class='prediction-value'>{pred_display}</div>
              <div class='prediction-note'>{note}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: AUTOML ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 🤖 AutoML — Train a Prediction Model")
        st.caption("Select a target column. Trains 3 models automatically and picks the best one.")

        all_cols = list(df_clean.columns)
        target_col = st.selectbox("Select target column to predict:", all_cols)

        col_info = df_clean[target_col]
        if col_info.dtype == object or col_info.nunique() <= 10:
            st.info(f"📌 Detected task: **Classification** ({col_info.nunique()} classes)")
        else:
            st.info(f"📌 Detected task: **Regression** (numeric target)")

        if st.button("🏋️ Train Model", type="primary", use_container_width=True):
            with st.spinner("Training 3 models and selecting the best..."):
                model_result = train_model(df_clean, target_col)
                if model_result["status"] == "error":
                    st.error(model_result["message"])
                else:
                    st.session_state.model_result = model_result
                    st.success(f"✅ Best model: **{model_result['best_model']}** — go to What-If Simulator to explore it!")

        if st.session_state.model_result:
            mr = st.session_state.model_result

            st.markdown("---")
            st.markdown("#### 📊 Model Performance")

            if mr["task"] == "classification":
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><div class='value'>{mr['accuracy']}%</div><div class='label'>Accuracy</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><div class='value'>{mr['best_model'].split()[0]}</div><div class='label'>Best Model</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><div class='value'>{mr['test_size']}</div><div class='label'>Test Samples</div></div>", unsafe_allow_html=True)
                if "classification_report" in mr:
                    st.markdown("#### Classification Report")
                    st.dataframe(pd.DataFrame(mr["classification_report"]).T.round(2), use_container_width=True)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"<div class='metric-card'><div class='value'>{mr['r2_score']}</div><div class='label'>R² Score</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><div class='value'>{mr['mae']}</div><div class='label'>MAE</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><div class='value'>{mr['rmse']}</div><div class='label'>RMSE</div></div>", unsafe_allow_html=True)
                c4.markdown(f"<div class='metric-card'><div class='value'>{mr['test_size']}</div><div class='label'>Test Samples</div></div>", unsafe_allow_html=True)

            if mr.get("charts"):
                st.markdown("#### Model Charts")
                cols2 = st.columns(2)
                for i, chart in enumerate(mr["charts"]):
                    with cols2[i % 2]:
                        st.markdown(f"**{chart['title']}**")
                        st.image(base64.b64decode(chart["image_base64"]), use_container_width=True)

            if mr.get("top_features"):
                st.markdown("#### 🏆 Top 5 Important Features")
                st.dataframe(pd.DataFrame(mr["top_features"]), use_container_width=True)


    # ── TAB 4: XAI EXPLAINER ──────────────────────────────────────────────────
    with tab4:
        st.markdown("#### 🔍 Explainable AI — SHAP Analysis")
        st.caption("Understand *why* the model makes each prediction. Based on SHAP (SHapley Additive exPlanations) — used by industry ML teams.")

        if st.session_state.model_result is None:
            st.info("👉 Go to the **AutoML** tab first, train a model, then come back here.")
        else:
            mr = st.session_state.model_result
            st.success(f"Explaining: **{mr['best_model']}** predicting **{mr['target_column']}**")

            if st.session_state.shap_result is None:
                if st.button("🔍 Generate SHAP Explanation", type="primary", use_container_width=True):
                    with st.spinner("Calculating SHAP values — this takes 20-30 seconds..."):
                        shap_result = explain_model(mr, df_clean)
                        st.session_state.shap_result = shap_result
                    if shap_result["status"] == "error":
                        st.error(shap_result["message"])
                        if "shap not installed" in shap_result["message"]:
                            st.code("pip install shap")
                    else:
                        st.success("✅ SHAP analysis complete!")
                        st.rerun()
            
            if st.session_state.shap_result and st.session_state.shap_result["status"] == "success":
                sr = st.session_state.shap_result

                # Top feature insights
                st.markdown("#### 💡 What the model learned")
                if sr.get("top_features"):
                    cols3 = st.columns(min(3, len(sr["top_features"])))
                    for i, feat in enumerate(sr["top_features"][:3]):
                        with cols3[i]:
                            direction_color = "#22c55e" if feat["direction"] == "increases" else "#ef4444"
                            direction_arrow = "↑" if feat["direction"] == "increases" else "↓"
                            st.markdown(f"""
                            <div class='category-card'>
                              <div class='cat-name'>#{i+1} {feat['feature']}</div>
                              <div class='cat-score' style='color:{direction_color};font-size:18px;'>
                                {direction_arrow} Impact: {feat['mean_impact']}
                              </div>
                              <div class='cat-insight'>{feat['insight']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("")

                st.caption(f"Analysis based on {sr['sample_size']} samples · {sr['feature_count']} features")

                # SHAP Charts
                st.markdown("---")
                st.markdown("#### 📊 SHAP Visualisations")
                if sr.get("charts"):
                    for chart in sr["charts"]:
                        st.markdown(f"**{chart['title']}**")
                        st.image(base64.b64decode(chart["image_base64"]), use_container_width=True)
                        st.markdown("")

                # What is SHAP explainer box
                with st.expander("📖 What is SHAP? How to read these charts?"):
                    st.markdown("""
**SHAP (SHapley Additive exPlanations)** assigns each feature a score showing how much it contributed to a specific prediction.

**Feature Importance chart** — which features matter most on average across all predictions. Longer bar = more important.

**Beeswarm Plot** — each dot is one row in your dataset. Red dots = high feature value, Blue = low. Dots on the right = pushed prediction up. On the left = pushed it down.

**Single Prediction chart** — for one specific row, shows exactly which features pushed the prediction up (red) or down (blue).

**Why this matters:** A model might be 87% accurate, but without SHAP you don't know *why*. SHAP makes the black box transparent — critical for real-world deployment in healthcare, finance, and HR.
                    """)

                if st.button("🔄 Recalculate SHAP", key="reshap"):
                    st.session_state.shap_result = None
                    st.rerun()

    # ── TAB 5: CHAT ────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### 💬 Ask anything about your dataset")
        st.caption("Agent 2 answers using the full analysis as context — RAG pattern.")

        eda = results.get("eda", {})
        cols = eda.get("column_names", [])
        suggestions = [
            "What are the key insights?",
            "Which columns have most anomalies?",
            "What do the correlations mean?",
            "Which column to use as prediction target?",
        ]
        if cols:
            suggestions.append(f"Tell me about '{cols[0]}'")

        st.markdown("**💡 Quick questions:**")
        scols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if scols[i].button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    answer = st.session_state.chat_agent.ask(q)
                st.session_state.chat_history.append({"role": "ai", "content": answer})
                st.rerun()

        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown("<p style='color:#a0a0c0;font-size:14px;text-align:center;padding:20px;'>Ask a question above or type below 👇</p>", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"<div class='chat-label'>You</div><div class='chat-msg-user'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-label'>AI Analyst</div><div class='chat-msg-ai'>{msg['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Your question", placeholder="e.g. What is the average of column X?", label_visibility="collapsed")
            submitted = st.form_submit_button("Send →", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                answer = st.session_state.chat_agent.ask(user_input.strip())
            st.session_state.chat_history.append({"role": "ai", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.session_state.chat_agent.reset()
                st.rerun()

    # ── TAB 5: SUMMARY ─────────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### AI-Generated Summary")
        st.markdown(summary)
        if "eda" in results:
            eda = results["eda"]
            if "numeric_stats" in eda:
                st.markdown("#### Numeric Statistics")
                st.dataframe(pd.DataFrame(eda["numeric_stats"]), use_container_width=True)
            if eda.get("high_correlations"):
                st.markdown("#### High Correlations")
                st.dataframe(pd.DataFrame(eda["high_correlations"]), use_container_width=True)

    # ── TAB 6: CHARTS ──────────────────────────────────────────────────────────
    with tab6:
        if "charts" in results and results["charts"]["charts"]:
            charts = results["charts"]["charts"]
            cols2 = st.columns(2)
            for i, chart in enumerate(charts):
                with cols2[i % 2]:
                    st.markdown(f"**{chart['title']}**")
                    st.image(base64.b64decode(chart["image_base64"]), use_container_width=True)
        else:
            st.info("No charts generated.")

    # ── TAB 8: DOWNLOAD ────────────────────────────────────────────────────────
    with tab8:
        st.markdown("#### Download Full HTML Report")
        try:
            with open("report_template.html") as f:
                template_str = f.read()

            clean_info = results.get("clean_data", {})
            eda_info   = results.get("eda", {})
            out_info   = results.get("outliers", {})
            chart_info = results.get("charts", {})

            html_out = Template(template_str).render(
                generated_at=datetime.now().strftime("%d %b %Y, %H:%M"),
                filename=uploaded_file.name,
                rows=eda_info.get("rows", df_raw.shape[0]),
                columns=eda_info.get("columns", df_raw.shape[1]),
                duplicates_removed=clean_info.get("duplicates_removed", 0),
                missing_filled=clean_info.get("missing_values_filled", 0),
                dtypes=eda_info.get("dtypes", {}),
                outlier_columns=out_info.get("columns_with_outliers", []),
                outlier_details=out_info.get("outlier_details", {}),
                charts=chart_info.get("charts", []),
                summary=summary,
            )

            st.download_button(
                label="⬇️ Download Report (HTML)",
                data=html_out.encode("utf-8"),
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True,
            )
            st.caption("Open in any browser to view the full report.")
        except Exception as e:
            st.error(f"Could not generate report: {e}")
