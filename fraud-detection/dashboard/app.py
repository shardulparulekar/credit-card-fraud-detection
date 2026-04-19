"""
dashboard/app.py  —  Streamlit Dashboard
=========================================
Interactive fraud detection UI with filtering.

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("models/best_model.pkl"),
        joblib.load("models/scaler.pkl"),
        float(joblib.load("models/threshold.pkl")),
        joblib.load("models/model_name.pkl"),
    )

if not os.path.exists("models/best_model.pkl"):
    st.error("Run `python pipeline.py` first to train the model.")
    st.stop()

model, scaler, threshold, model_name = load_artifacts()

# ── Load real test set saved by pipeline.py ────────────────────────────────────
@st.cache_data
def get_scored_data():
    """
    Loads the real test split saved by pipeline.py.
    Falls back to synthetic data only if the real files don't exist yet.
    """
    if os.path.exists("data/X_test.parquet") and os.path.exists("data/y_test.parquet"):
        X_t = pd.read_parquet("data/X_test.parquet")
        y_t = pd.read_parquet("data/y_test.parquet")["Class"]

        # Reconstruct Amount and Time from scaled columns for display
        # (we store Amount_scaled and Time_scaled — invert with scaler)
        amount_time = scaler.inverse_transform(
            X_t[["Amount_scaled", "Time_scaled"]]
        )
        meta = pd.DataFrame({
            "Amount": amount_time[:, 0].round(2),
            "Time":   amount_time[:, 1].round(0),
        }, index=X_t.index)

    else:
        # Fallback: synthetic data (pipeline not yet run with real dataset)
        st.warning("Real test data not found — showing synthetic data. Re-run `python pipeline.py` with the Kaggle dataset.")
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler as SS
        np.random.seed(42)
        X, y = make_classification(n_samples=50_000, n_features=28, n_informative=15,
                                    n_redundant=5, weights=[0.98, 0.02], random_state=42, flip_y=0.01)
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
        df["Amount"] = np.random.lognormal(3.5, 1.5, 50_000).clip(0.01, 10_000).round(2)
        df["Time"]   = np.random.uniform(0, 172_800, 50_000).round(0)
        df["Class"]  = y
        sc = SS()
        df[["Amount_scaled", "Time_scaled"]] = sc.fit_transform(df[["Amount", "Time"]])
        X_ = df.drop(columns=["Class", "Amount", "Time"])
        y_ = df["Class"]
        meta_ = df[["Amount", "Time"]]
        _, X_t, _, y_t, _, meta = train_test_split(X_, y_, meta_, test_size=0.30, stratify=y_, random_state=42)
        X_t, _, y_t, _, meta, _ = train_test_split(X_t, y_t, meta, test_size=0.50, stratify=y_t, random_state=42)
        X_t = X_t.reset_index(drop=True)
        y_t = y_t.reset_index(drop=True)
        meta = meta.reset_index(drop=True)

    # Score the test set
    probs = model.predict_proba(X_t)[:, 1]
    preds = (probs >= threshold).astype(int)

    scored = meta.copy()
    scored["fraud_probability"] = probs.round(4)
    scored["prediction"]        = np.where(preds, "Fraud", "Legit")
    scored["risk_level"]        = pd.cut(probs, [0, .3, .6, 1.],
                                          labels=["LOW", "MEDIUM", "HIGH"]).astype(str)
    scored["actual"]            = y_t.map({0: "Legit", 1: "Fraud"})
    scored["correct"]           = (preds == y_t.values)
    scored["time_hours"]        = (scored["Time"] / 3600).round(1)

    return X_t, y_t, scored, probs

X_test, y_test, scored_df, all_probs = get_scored_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💳 Payment Fraud Detection")
st.caption(f"Model: **{model_name}**  ·  Threshold: **{threshold:.3f}**  ·  Test set: **{len(X_test):,} transactions**")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Transaction Explorer", "📁 Batch Scoring"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  average_precision_score, roc_auc_score, confusion_matrix)
    preds_all = (all_probs >= threshold).astype(int)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PR-AUC",    f"{average_precision_score(y_test, all_probs):.3f}")
    c2.metric("ROC-AUC",   f"{roc_auc_score(y_test, all_probs):.3f}")
    c3.metric("Recall",    f"{recall_score(y_test, preds_all):.3f}",    help="% of actual fraud caught")
    c4.metric("Precision", f"{precision_score(y_test, preds_all):.3f}", help="% of alerts that are real fraud")
    c5.metric("F1",        f"{f1_score(y_test, preds_all):.3f}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Fraud Score Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(all_probs[y_test == 0], bins=50, alpha=0.6, color="#2ecc71", label="Legit")
        ax.hist(all_probs[y_test == 1], bins=50, alpha=0.8, color="#e74c3c", label="Fraud")
        ax.axvline(threshold, color="navy", linestyle="--", label=f"Threshold = {threshold:.2f}")
        ax.set_xlabel("Fraud Probability"); ax.legend()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, preds_all)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Pred Legit", "Pred Fraud"])
        ax2.set_yticklabels(["Actual Legit", "Actual Fraud"])
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    if os.path.exists("outputs/pr_curves.png"):
        st.divider()
        st.image("outputs/pr_curves.png", caption="Precision-Recall curves — all models")
    if os.path.exists("outputs/shap_bar.png"):
        st.image("outputs/shap_bar.png", caption="Top fraud features (SHAP)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Transaction Explorer with Filters
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Transaction Explorer")
    st.markdown("Use the filters below to zero in on suspicious transactions.")

    with st.expander("🎛  Filters", expanded=True):
        f1, f2, f3 = st.columns(3)

        with f1:
            st.markdown("**Prediction**")
            show_fraud = st.checkbox("🚨 Flagged as Fraud", value=True)
            show_legit = st.checkbox("✅ Flagged as Legit",  value=False)

        with f2:
            st.markdown("**Risk Level**")
            risk_high   = st.checkbox("🔴 HIGH",   value=True)
            risk_medium = st.checkbox("🟡 MEDIUM", value=True)
            risk_low    = st.checkbox("🟢 LOW",    value=True)

        with f3:
            st.markdown("**Fraud Probability**")
            prob_min, prob_max = st.slider("Range", 0.0, 1.0, (0.0, 1.0), step=0.01)

        fa, fb = st.columns(2)
        with fa:
            st.markdown("**Amount (USD)**")
            real_max_amt = float(scored_df["Amount"].max())
            amt_min, amt_max = st.slider("Amount range", 0.0, real_max_amt,
                                          (0.0, real_max_amt), step=1.0)
        with fb:
            st.markdown("**Time (hours since start)**")
            time_min, time_max = st.slider("Time range (hrs)", 0.0, 48.0, (0.0, 48.0), step=0.5)

    # Apply filters
    selected_predictions = []
    if show_fraud: selected_predictions.append("Fraud")
    if show_legit: selected_predictions.append("Legit")

    selected_risks = []
    if risk_high:   selected_risks.append("HIGH")
    if risk_medium: selected_risks.append("MEDIUM")
    if risk_low:    selected_risks.append("LOW")

    filtered = scored_df[
        (scored_df["prediction"].isin(selected_predictions)) &
        (scored_df["risk_level"].isin(selected_risks)) &
        (scored_df["fraud_probability"].between(prob_min, prob_max)) &
        (scored_df["Amount"].between(amt_min, amt_max)) &
        (scored_df["time_hours"].between(time_min, time_max))
    ].copy()

    # Summary metrics
    total    = len(filtered)
    n_fraud  = (filtered["prediction"] == "Fraud").sum()
    avg_prob = filtered["fraud_probability"].mean() if total > 0 else 0
    avg_amt  = filtered["Amount"].mean() if total > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching Transactions", f"{total:,}")
    m2.metric("Flagged as Fraud",       f"{n_fraud:,}")
    m3.metric("Avg Fraud Probability",  f"{avg_prob:.3f}")
    m4.metric("Avg Amount",             f"${avg_amt:.2f}")

    st.divider()

    if total == 0:
        st.warning("No transactions match your filters — try relaxing them.")
    else:
        display_cols = ["Amount", "time_hours", "fraud_probability",
                        "risk_level", "prediction", "actual", "correct"]
        display = (
            filtered[display_cols]
            .rename(columns={
                "time_hours":        "Time (hrs)",
                "fraud_probability": "Fraud Prob",
                "risk_level":        "Risk",
                "prediction":        "Predicted",
                "actual":            "Actual",
                "correct":           "Correct?",
            })
            .sort_values("Fraud Prob", ascending=False)
            .head(200)
        )

        st.dataframe(
            display.style
                .background_gradient(subset=["Fraud Prob"], cmap="RdYlGn_r", vmin=0, vmax=1)
                .format({"Fraud Prob": "{:.4f}", "Amount": "${:.2f}", "Time (hrs)": "{:.1f}"}),
            use_container_width=True,
            height=360,
        )
        st.caption(f"Showing top 200 of {total:,} matching transactions, sorted by fraud probability.")

        # SHAP explanation
        st.divider()
        st.markdown("**🧠 Explain a Transaction**")
        st.caption("Enter a row index from the table above to see why the model made that prediction.")

        default_idx = int(filtered.index[0]) if total > 0 else int(scored_df.index[0])
        min_idx = int(scored_df.index.min())
        max_idx = int(scored_df.index.max())
        idx_input = st.number_input(
            "Transaction index", min_value=min_idx, max_value=max_idx,
            value=default_idx, step=1
        )

        if st.button("Generate SHAP Explanation"):
            # Use .loc to look up by label index, .iloc for the model feature row
            iloc_pos = scored_df.index.get_loc(idx_input)
            txn = X_test.iloc[[iloc_pos]]
            row = scored_df.loc[idx_input]

            ca, cb, cc, cd = st.columns(4)
            ca.metric("Amount",            f"${row['Amount']:.2f}")
            cb.metric("Fraud Probability", f"{row['fraud_probability']:.4f}")
            cc.metric("Predicted",         f"{'🚨 Fraud' if row['prediction']=='Fraud' else '✅ Legit'}")
            cd.metric("Actual",            row["actual"])

            with st.spinner("Computing SHAP values..."):
                try:
                    import shap
                    exp     = shap.TreeExplainer(model)
                    sv      = exp.shap_values(txn)
                    sv      = sv[1] if isinstance(sv, list) else sv
                    contrib = (
                        pd.Series(sv[0], index=txn.columns)
                        .sort_values(key=abs, ascending=False)
                        .head(12)
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib.values]
                    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_xlabel("SHAP value  (red = toward fraud  |  green = toward legit)")
                    ax.set_title(f"Transaction #{idx_input}  →  Predicted: {row['prediction']}  |  Actual: {row['actual']}")
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

                    top       = contrib.abs().idxmax()
                    direction = "toward fraud" if contrib[top] > 0 else "away from fraud"
                    st.info(f"💡 Strongest signal: **{top}** — pushed prediction {direction}.")
                except Exception as e:
                    st.error(str(e))

        st.divider()
        csv = filtered[display_cols].to_csv(index=True)
        st.download_button("⬇ Download filtered results as CSV", csv,
                           "filtered_transactions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Scoring
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Batch Scoring")
    n = st.slider("Transactions to score", 10, 500, 100)

    if st.button("▶ Score Batch"):
        sample  = X_test.sample(n, random_state=np.random.randint(9999))
        sp      = model.predict_proba(sample)[:, 1]
        sp_pred = (sp >= threshold).astype(int)
        out = pd.DataFrame({
            "fraud_probability": sp.round(4),
            "prediction":  np.where(sp_pred, "🚨 FRAUD", "✅ LEGIT"),
            "risk_level":  pd.cut(sp, [0, .3, .6, 1.],
                                   labels=["LOW", "MEDIUM", "HIGH"]).astype(str)
        })
        n_flagged = sp_pred.sum()
        st.success(f"**{n_flagged} of {n}** flagged as fraud ({n_flagged/n*100:.1f}%)")
        st.dataframe(
            out.sort_values("fraud_probability", ascending=False)
               .style.background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r", vmin=0, vmax=1)
               .format({"fraud_probability": "{:.4f}"}),
            use_container_width=True
        )
        st.download_button("⬇ Download CSV", out.to_csv(index=True),
                           "fraud_scores.csv", "text/csv")

st.divider()
st.caption("Built with XGBoost / LightGBM · SHAP · FastAPI · Streamlit")
