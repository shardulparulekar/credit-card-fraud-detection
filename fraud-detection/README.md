# 💳 Payment Fraud Detection

> An end-to-end machine learning system for real-time payment fraud detection — featuring automatic data download, model training, explainability, a REST API, and an interactive dashboard.

---

## 📌 Overview

Payment fraud costs the global economy **$40+ billion annually**. Rule-based systems (e.g. "flag all transactions over $1,000") are rigid and easy to bypass. This project builds a **machine learning pipeline** that learns subtle fraud patterns from historical data — catching fraud that rules miss while minimising false alarms.

### Key Features
- **Auto-downloads** the real Kaggle dataset on first run via the Kaggle API
- **3 models compared** — Logistic Regression, XGBoost, LightGBM
- **Class imbalance handled** — SMOTE oversampling + threshold tuning
- **Explainability** — SHAP values show *why* each transaction was flagged
- **REST API** — FastAPI endpoint for real-time scoring
- **Interactive dashboard** — Streamlit UI for exploration and batch scoring
- **One command to run** — Docker handles everything

---

## 🗂 Project Structure

```
fraud-detection/
│
├── pipeline.py          ← Full ML pipeline: download → train → explain
├── download_data.py     ← Kaggle API dataset downloader
├── api/
│   └── main.py          ← FastAPI REST endpoint
├── dashboard/
│   └── app.py           ← Streamlit interactive dashboard
│
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### Step 1 — Get your Kaggle API token

1. Go to [kaggle.com](https://kaggle.com) → Profile → **Settings** → **API** → **Generate New Token**
2. Copy the token string shown (starts with `KGAT_...`) — you only see it once
3. Accept the dataset rules at: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) *(required once)*

### Step 2 — Export your token

```bash
export KAGGLE_API_TOKEN=KGAT_your_token_here
```

### Step 3 — Run (single command)

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
docker compose up --build
```

Docker will automatically:
1. Download the real Credit Card Fraud dataset from Kaggle (~150 MB)
2. Train and compare Logistic Regression, XGBoost, and LightGBM
3. Run SHAP explainability analysis
4. Launch the dashboard at **http://localhost:8501**
5. Launch the REST API at **http://localhost:8000/docs**

> The dataset and trained model are cached in `./data/` and `./models/` via Docker volumes — so subsequent `docker compose up` runs skip the download and training.

---

### Running without Docker

```bash
pip install -r requirements.txt

# Set token, then run
export KAGGLE_API_TOKEN=KGAT_your_token_here
python pipeline.py

# Launch dashboard
streamlit run dashboard/app.py

# Launch API (separate terminal)
uvicorn api.main:app --reload --port 8000
```

> **No Kaggle account?** Run `python pipeline.py --no-download` to use synthetic data instead.
> **Lost your token?** Just generate a new one — kaggle.com → Profile → Settings → API.

---

## 📊 ML Approach

### Why not accuracy?

With only 0.17% of transactions being fraudulent, a model predicting "legitimate" for everything achieves **99.83% accuracy** — but catches zero fraud. We use instead:

| Metric | What it measures |
|---|---|
| **PR-AUC** | Overall model quality on imbalanced data *(primary metric)* |
| **Recall** | % of actual fraud caught — missing fraud is costly |
| **Precision** | % of flagged transactions that are real fraud |
| **F1** | Harmonic mean of recall and precision |

### Handling class imbalance

| Technique | Why |
|---|---|
| **SMOTE** | Synthesises new fraud examples so the model sees enough to learn from |
| **Class weights** | Penalises missing fraud more than false alarms |
| **Threshold tuning** | Adjusts the decision cut-off beyond the default 0.5 to maximise F1 |

### Models compared

| Model | Role |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| XGBoost | Gradient boosting — handles non-linear patterns |
| LightGBM | Fast gradient boosting — efficient on large datasets |

The best model by PR-AUC on the validation set is automatically selected and saved.

### Explainability — SHAP

Every prediction is explained using **SHAP (SHapley Additive exPlanations)**:
- Which features pushed this transaction toward fraud?
- Which features indicated it was legitimate?

This matters in regulated industries like payments, where automated decisions must be justifiable to auditors and regulators.

---

## 🌐 API Reference

### `POST /predict` — Score a single transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 80000, "Amount": 149.62, "V1": -1.36, "V2": -0.07}'
```

```json
{
  "fraud_probability": 0.8732,
  "is_fraud": true,
  "risk_level": "HIGH",
  "model_used": "XGBoost",
  "threshold_used": 0.42
}
```

### `POST /predict/batch` — Score up to 100 transactions at once

Full interactive docs at **http://localhost:8000/docs**

---

## 📦 Dataset

**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** — Worldline & ULB Machine Learning Group

| Property | Value |
|---|---|
| Rows | 284,807 transactions |
| Fraud rate | 0.172%  (492 fraud out of 284k) |
| Features | V1–V28 (PCA-anonymised) + Amount + Time |
| Academic citations | 1,000+ |

The V1–V28 features are real transaction attributes transformed via PCA to protect cardholder privacy — exactly what you'd encounter working with real payment data at a fintech company.

The dataset is **not stored in this repository**. It is downloaded automatically from Kaggle on first run using the Kaggle API.

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn |
| Boosting models | XGBoost, LightGBM |
| Class imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| REST API | FastAPI, uvicorn |
| Dashboard | Streamlit |
| Visualisation | matplotlib |
| Deployment | Docker, docker compose |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built as a portfolio project for an MBA in AI — demonstrating end-to-end ML engineering applied to a real fintech problem.*
