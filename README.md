<div align="center">

# 📊 DataMind AI

### Intelligent Data Analysis & AutoML Platform

**A production-grade, multi-agent AI system that transforms raw datasets into actionable insights — automatically.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Mistral_7B-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-AutoML-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green?style=flat-square)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

[Live Demo](#) · [Report Bug](issues) · [Features](#features)

</div>

---

## 🧠 What is DataMind AI?

DataMind AI is an end-to-end intelligent data analysis platform powered by a **multi-agent AI architecture**. Upload any CSV or Excel dataset and the system autonomously:

- **Cleans** your data using a dedicated Analyst Agent
- **Analyses** patterns, correlations, and distributions
- **Scores** your dataset quality across 6 dimensions
- **Trains** machine learning models automatically (AutoML)
- **Explains** every prediction using SHAP (Explainable AI)
- **Simulates** what-if scenarios with live predictions
- **Answers** your questions in plain English via a RAG-based Chat Agent
- **Generates** a downloadable professional HTML report

No code. No configuration. Just upload and go.

---

## ✨ Features

### 🤖 Multi-Agent Architecture

Two specialised AI agents working in parallel:

- **Agent 1 — Analyst**: Orchestrates the full data pipeline (clean → EDA → outliers → charts → report)
- **Agent 2 — Chat (RAG)**: Answers natural language questions grounded in the analysis results

### 🏆 Data Quality Scorecard

Automated dataset health check across 6 dimensions:
| Dimension | What it checks |
|---|---|
| Completeness | Missing values |
| Uniqueness | Duplicate rows |
| Consistency | Type & format issues |
| Validity | Outliers (IQR + Z-score) |
| Distribution | Skewness |
| Balance | Class imbalance |

Each dimension gets a score (0–100) and grade (A–F), with a weighted overall score and actionable recommendations.

### 🤖 AutoML Engine

Trains and evaluates 3 models automatically:

- Random Forest
- Gradient Boosting
- Logistic / Linear Regression

Auto-detects classification vs regression. Picks the best model. Shows accuracy, R², confusion matrix, feature importance.

### 🔍 Explainable AI (SHAP)

Post-hoc model explainability using SHAP values:

- **Feature Importance chart** — which features matter most
- **Beeswarm Plot** — how feature values affect predictions across the dataset
- **Single Prediction Explanation** — why the model made a specific decision
- Plain-English insight cards: _"Higher Age increases predicted Risk"_

### 🎯 What-If Scenario Simulator

Interactive sliders and dropdowns for every feature. Move a slider → prediction updates instantly. Enables real-time decision simulation without writing a single line of code.

### 💬 RAG-Based Chat Agent

Ask anything about your dataset in plain English:

> _"Which column has the most outliers?"_
> _"What does the correlation between Age and Income mean?"_
> _"Which feature should I focus on for prediction?"_

Multi-turn conversation with full history. Powered by Mistral-7B via HuggingFace Inference API.

### 📥 Downloadable HTML Report

Professional report with dataset overview, statistics, outlier analysis, charts, and AI summary. Opens in any browser — no dependencies.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│   Upload → Analyse → Score → Train → Chat        │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌────────▼────────┐
│  Agent 1       │      │  Agent 2        │
│  Analyst       │      │  Chat (RAG)     │
│                │      │                 │
│ ┌────────────┐ │      │ Context: EDA +  │
│ │ clean_data │ │      │ stats + summary  │
│ │ run_eda    │ │      │                 │
│ │ detect_out │ │      │ Mistral-7B via  │
│ │ make_charts│ │      │ HuggingFace API │
│ └────────────┘ │      └─────────────────┘
│                │
│ Mistral-7B     │
│ (summary gen)  │
└───────┬────────┘
        │
┌───────▼────────────────────────────────────────┐
│              AutoML Engine                      │
│  Random Forest + Gradient Boosting + LR/Logit  │
│  Auto task detection · Best model selection    │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│           SHAP Explainability Layer             │
│  TreeExplainer · Beeswarm · Force plots        │
└────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer             | Technology                                        |
| ----------------- | ------------------------------------------------- |
| Frontend          | Streamlit                                         |
| LLM               | Mistral-7B-Instruct via HuggingFace Inference API |
| Data Processing   | pandas, numpy, scipy                              |
| Machine Learning  | scikit-learn                                      |
| Explainability    | SHAP                                              |
| Visualisation     | matplotlib, seaborn                               |
| Report Generation | Jinja2 HTML templates                             |
| Deployment        | Streamlit Community Cloud                         |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda or virtualenv
- Free HuggingFace account ([sign up here](https://huggingface.co/join))

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/nidhiv1303/ai-data-analyst.git
cd ai-data-analyst
```

**2. Create and activate conda environment**

```bash
conda create -n datamind python=3.12
conda activate datamind
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your API key**

```bash
cp .env.example .env
```

Open `.env` and paste your HuggingFace token:

```
HF_API_KEY=hf_your_token_here
```

Get your free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**5. Run the app**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
ai_data_analyst/
├── app.py                      # Main Streamlit application
├── agent.py                    # Agent 1 — Analyst (tool-calling loop)
├── chat_agent.py               # Agent 2 — Chat (RAG conversation)
├── report_template.html        # Jinja2 HTML report template
├── requirements.txt            # Python dependencies
├── .env.example                # API key template
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
└── tools/
    ├── __init__.py
    ├── clean_data.py           # Data cleaning tool
    ├── run_eda.py              # Exploratory data analysis tool
    ├── detect_outliers.py      # IQR + Z-score outlier detection
    ├── make_charts.py          # Chart generation (matplotlib/seaborn)
    ├── automl.py               # AutoML training + prediction
    ├── data_quality.py         # 6-dimension quality scoring
    └── explainability.py       # SHAP explainability engine
```

---

## 📊 Demo

**Upload any dataset** → the system handles everything automatically.

Try it with these free datasets:

- [Titanic](https://www.kaggle.com/datasets/yasserh/titanic-dataset) — classification
- [House Prices](https://www.kaggle.com/datasets/lespin/house-prices-dataset) — regression
- [German Credit](https://www.kaggle.com/datasets/uciml/german-credit) — classification

---

## 🎯 Use Cases

- **Data Scientists** — rapid EDA and baseline model benchmarking
- **Business Analysts** — understand datasets without writing code
- **Students** — learn ML concepts through interactive visualisations
- **Researchers** — quick dataset profiling before deep analysis

---

## 🔮 Roadmap

- [ ] Natural Language to Pandas query engine
- [ ] Model drift detection (upload new data, check performance degradation)
- [ ] Automated feature engineering
- [ ] SQL database support
- [ ] React + FastAPI frontend

---

## 👤 Author

**Your Name**

- GitHub: [@nidhiv1303](https://github.com/nidhiv1303)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <b>If this project helped you, please ⭐ star the repo!</b>
</div>
