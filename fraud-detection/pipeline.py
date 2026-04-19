"""
pipeline.py
===========
Complete fraud detection pipeline in a single file.
Runs data download → preprocessing → training → evaluation → SHAP explainability.

Usage:
    python pipeline.py                  # auto-downloads real Kaggle dataset
    python pipeline.py --no-download    # skip download, use synthetic data instead

Kaggle credentials required for download (one-time setup):
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
  OR place kaggle.json at ~/.config/kaggle/kaggle.json
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples=50_000):
    """Creates a realistic synthetic fraud dataset when no real CSV is provided."""
    print("  Generating synthetic dataset...")
    np.random.seed(42)

    X, y = make_classification(
        n_samples=n_samples, n_features=28, n_informative=15,
        n_redundant=5, weights=[0.98, 0.02], random_state=42, flip_y=0.01
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"]  = np.random.uniform(0, 172_800, size=n_samples)
    df["Amount"] = np.where(
        y == 1,
        np.where(np.random.rand(n_samples) > 0.5,
                 np.random.uniform(1, 50, n_samples),
                 np.random.uniform(500, 5_000, n_samples)),
        np.random.lognormal(3.5, 1.5, n_samples)
    ).clip(0.01, 10_000).round(2)
    df["Class"] = y
    return df


def load_data(csv_path=None):
    print("\n" + "═" * 55)
    print("  STEP 1/4 · Loading Data")
    print("═" * 55)

    if csv_path and os.path.exists(csv_path):
        print(f"  Loading real dataset: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        if csv_path:
            print(f"  ⚠  File not found: {csv_path}. Falling back to synthetic data.")
        df = generate_synthetic_data()

    print(f"  Rows       : {len(df):,}")
    print(f"  Fraud rate : {df['Class'].mean()*100:.3f}%  ({df['Class'].sum():,} fraud txns)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df):
    print("\n" + "═" * 55)
    print("  STEP 2/4 · Preprocessing")
    print("═" * 55)

    scaler = StandardScaler()
    df[["Amount_scaled", "Time_scaled"]] = scaler.fit_transform(df[["Amount", "Time"]])
    df = df.drop(columns=["Amount", "Time"])
    joblib.dump(scaler, "models/scaler.pkl")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"  Train : {len(X_train):,} rows  |  fraud: {y_train.sum():,}")
    print(f"  Val   : {len(X_val):,} rows  |  fraud: {y_val.sum():,}")
    print(f"  Test  : {len(X_test):,} rows  |  fraud: {y_test.sum():,}")
    print(f"  Scaler saved → models/scaler.pkl")

    # Apply SMOTE only to training set
    print("\n  Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {y_train.sum():,} fraud / {len(y_train):,} total")
    print(f"  After  SMOTE: {y_res.sum():,} fraud / {len(y_res):,} total")

    # Save test set so the dashboard loads the real data (not synthetic)
    os.makedirs("data", exist_ok=True)
    X_test.to_parquet("data/X_test.parquet", index=True)
    y_test.to_frame(name="Class").to_parquet("data/y_test.parquet", index=True)
    print(f"  Test set saved → data/X_test.parquet")

    return X_res, y_res, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1)
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def evaluate(name, model, X, y, threshold, label=""):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    pr_auc = average_precision_score(y, probs)
    roc    = roc_auc_score(y, probs)
    f1     = f1_score(y, preds)
    print(f"\n  {'─'*45}")
    print(f"  {name}  {label}")
    print(f"  PR-AUC : {pr_auc:.4f}   ROC-AUC : {roc:.4f}   F1 : {f1:.4f}")
    report = classification_report(y, preds, target_names=["Legit", "Fraud"])
    for line in report.splitlines():
        print(f"    {line}")
    return pr_auc, roc, f1


def train(X_res, y_res, X_val, y_val, X_test, y_test):
    print("\n" + "═" * 55)
    print("  STEP 3/4 · Training Models")
    print("═" * 55)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "XGBoost":  XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   scale_pos_weight=10, eval_metric="aucpr",
                                   random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    class_weight="balanced", random_state=42, verbose=-1),
    }

    results, probs_dict = {}, {}
    best_name, best_pr = None, 0

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_res, y_res)
        threshold = find_best_threshold(model, X_val, y_val)
        pr_auc, roc, f1 = evaluate(name, model, X_val, y_val, threshold, "(validation)")
        results[name] = dict(model=model, threshold=threshold, pr_auc=pr_auc)
        probs_dict[name] = model.predict_proba(X_val)[:, 1]
        if pr_auc > best_pr:
            best_pr, best_name = pr_auc, name

    # PR curve comparison plot
    plt.figure(figsize=(8, 5))
    for name, probs in probs_dict.items():
        p, r, _ = precision_recall_curve(y_val, probs)
        auc = average_precision_score(y_val, probs)
        plt.plot(r, p, label=f"{name}  (PR-AUC={auc:.3f})", linewidth=2)
    plt.xlabel("Recall  (% of fraud caught)");  plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — All Models");  plt.legend();  plt.tight_layout()
    plt.savefig("outputs/pr_curves.png", dpi=150);  plt.close()
    print("\n  Chart saved → outputs/pr_curves.png")

    # Final evaluation on held-out test set
    best = results[best_name]
    print(f"\n  🏆 Best model: {best_name}  (PR-AUC = {best_pr:.4f})")
    evaluate(best_name, best["model"], X_test, y_test, best["threshold"], "(TEST SET — final)")

    # Persist
    joblib.dump(best["model"],     "models/best_model.pkl")
    joblib.dump(best["threshold"], "models/threshold.pkl")
    joblib.dump(best_name,         "models/model_name.pkl")
    print(f"\n  Model saved    → models/best_model.pkl")
    print(f"  Threshold saved → models/threshold.pkl")

    return best["model"], best["threshold"], X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def explain(model, X_test):
    print("\n" + "═" * 55)
    print("  STEP 4/4 · SHAP Explainability")
    print("═" * 55)

    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    print(f"  Computing SHAP values on {len(sample)} samples...")

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Bar summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, sample, plot_type="bar", show=False)
    plt.title("Top Features Driving Fraud Predictions")
    plt.tight_layout()
    plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight");  plt.close()
    print("  Chart saved → outputs/shap_bar.png")

    # Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, sample, show=False)
    plt.title("SHAP Values — Direction & Magnitude per Feature")
    plt.tight_layout()
    plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight");  plt.close()
    print("  Chart saved → outputs/shap_beeswarm.png")

    # Top feature summary in console
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top5 = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False).head(5)
    print("\n  Top 5 most important features:")
    for feat, val in top5.items():
        print(f"    {feat:25s} {val:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip Kaggle download and use synthetic data instead")
    args = parser.parse_args()

    print("\n🚀  Payment Fraud Detection — ML Pipeline")

    # ── Auto-download real dataset from Kaggle (default behaviour) ────────────
    kaggle_csv = "data/creditcard.csv"
    if args.no_download:
        data_path = None   # will trigger synthetic data generation
    else:
        try:
            from download_data import download_dataset
            download_dataset()          # no-op if file already exists
            data_path = kaggle_csv
        except EnvironmentError as e:
            print(f"\n  ⚠  {e}")
            print("  Falling back to synthetic data for this run.\n")
            data_path = None
        except Exception as e:
            print(f"\n  ⚠  Download failed: {e}")
            print("  Falling back to synthetic data for this run.\n")
            data_path = None

    df                                          = load_data(data_path)
    X_res, y_res, X_val, y_val, X_test, y_test = preprocess(df)
    model, threshold, X_test, y_test            = train(X_res, y_res, X_val, y_val, X_test, y_test)
    explain(model, X_test)

    print("\n" + "═" * 55)
    print("  ✅  Pipeline complete!")
    print("═" * 55)
    print("  Outputs:")
    print("    models/best_model.pkl       trained model")
    print("    outputs/pr_curves.png       model comparison chart")
    print("    outputs/shap_bar.png        feature importance")
    print("    outputs/shap_beeswarm.png   SHAP detail plot")
    print("\n  Next:")
    print("    streamlit run dashboard/app.py     launch dashboard")
    print("    uvicorn api.main:app --reload       launch REST API")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
