"""
ICU Mortality Prediction Dashboard
Stacked Ensemble: RF + XGBoost + CatBoost + LR → Logistic Meta Learner
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib
import shap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, roc_auc_score, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Mortality Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border-left: 4px solid #4f8ef7;
    }
    .metric-label { font-size: 13px; color: #555; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .risk-high { color: #d62728; font-weight: 700; font-size: 22px; }
    .risk-low  { color: #2ca02c; font-weight: 700; font-size: 22px; }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4f8ef7;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Loaders (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf        = joblib.load("rf.pkl")
    xgb       = joblib.load("xgb.pkl")
    cat       = joblib.load("cat.pkl")
    lr        = joblib.load("lr.pkl")
    meta      = joblib.load("meta.pkl")
    calibrator= joblib.load("calibrator.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    features  = joblib.load("feature_columns.pkl")
    t_cost    = joblib.load("threshold_cost.pkl")
    t_f1      = joblib.load("threshold_f1.pkl")
    return rf, xgb, cat, lr, meta, calibrator, explainer, features, t_cost, t_f1

@st.cache_data
def load_data():
    return pd.read_csv("final_merged_cleaned_preprocessed.csv")


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────
def predict_patient(patient_row, rf, xgb, cat, lr, meta, calibrator):
    rf_p   = rf.predict_proba(patient_row)[:, 1]
    xgb_p  = xgb.predict_proba(patient_row)[:, 1]
    cat_p  = cat.predict_proba(patient_row)[:, 1]
    lr_p   = lr.predict_proba(patient_row)[:, 1]
    meta_input = pd.DataFrame({"rf": rf_p, "xgb": xgb_p, "cat": cat_p, "lr": lr_p})
    raw_prob   = meta.predict_proba(meta_input)[:, 1][0]
    cal_prob   = float(calibrator.transform([raw_prob])[0])
    return cal_prob, {"RF": float(rf_p[0]), "XGBoost": float(xgb_p[0]),
                      "CatBoost": float(cat_p[0]), "LR": float(lr_p[0]), "Meta (raw)": float(raw_prob)}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("🏥 ICU Mortality")
st.sidebar.caption("Stacked Ensemble · RF + XGB + CaB + LR")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Performance", "🔬 Patient Risk Prediction", "📈 Feature Importance"],
    index=0,
)

# Attempt to load models
try:
    rf, xgb, cat, lr, meta, calibrator, explainer, features, t_cost, t_f1 = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)

# Attempt to load data
try:
    df = load_data()
    target = "bad_outcome"
    drop_cols = ["patientunitstayid", target]
    X = df.drop(columns=drop_cols)
    y = df[target]
    data_loaded = True
except Exception as e:
    data_loaded = False
    data_error = str(e)


# ─────────────────────────────────────────────
# Helper: show setup instructions if files missing
# ─────────────────────────────────────────────
def show_setup_warning(missing):
    st.warning(f"**Missing files:** {missing}")
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
**Run the following at the end of your notebook to save all required artifacts:**

```python
import joblib

joblib.dump(rf_final,    "rf.pkl")
joblib.dump(xgb_final,   "xgb.pkl")
joblib.dump(cat_final,   "cat.pkl")
joblib.dump(lr_final,    "lr.pkl")
joblib.dump(meta_final,  "meta.pkl")
joblib.dump(iso,         "calibrator.pkl")
joblib.dump(explainer,   "shap_explainer.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
joblib.dump(t_cost,      "threshold_cost.pkl")
joblib.dump(t_f1,        "threshold_f1.pkl")
```

Then place all `.pkl` files and `final_merged_cleaned_preprocessed.csv` in the
**same folder as `app.py`** and run:

```bash
streamlit run app.py
```
        """)


# ═══════════════════════════════════════════════════════════
# PAGE 1 — Model Performance
# ═══════════════════════════════════════════════════════════
if page == "📊 Model Performance":
    st.title("📊 Model Performance Overview")

    if not data_loaded or not models_loaded:
        show_setup_warning(
            ("data file" if not data_loaded else "") +
            (", model files" if not models_loaded else "")
        )
        st.stop()

    # ── Generate predictions ──────────────────────────
    with st.spinner("Generating out-of-fold predictions…"):
        rf_p   = rf.predict_proba(X)[:, 1]
        xgb_p  = xgb.predict_proba(X)[:, 1]
        cat_p  = cat.predict_proba(X)[:, 1]
        lr_p   = lr.predict_proba(X)[:, 1]
        meta_input = pd.DataFrame({"rf": rf_p, "xgb": xgb_p, "cat": cat_p, "lr": lr_p})
        meta_raw   = meta.predict_proba(meta_input)[:, 1]
        calibrated = calibrator.transform(meta_raw)

    # ── Threshold selector ────────────────────────────
    st.sidebar.markdown("---")
    threshold_mode = st.sidebar.selectbox(
        "Decision threshold",
        ["Safety (Cost 9:1)", "Balanced (F1)", "Custom"],
        index=0,
    )
    if threshold_mode == "Safety (Cost 9:1)":
        threshold = t_cost
    elif threshold_mode == "Balanced (F1)":
        threshold = t_f1
    else:
        threshold = st.sidebar.slider("Custom threshold", 0.01, 0.99, 0.30, 0.01)

    st.sidebar.caption(f"Active threshold: **{threshold:.3f}**")

    # ── Top metrics row ───────────────────────────────
    preds = (calibrated >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    auroc = roc_auc_score(y, calibrated)
    f1    = f1_score(y, preds)
    sens  = tp / (tp + fn)
    spec  = tn / (tn + fp)
    brier = brier_score_loss(y, calibrated)
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    def metric_card(col, label, value):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    metric_card(c1, "AUROC",       f"{auroc:.3f}")
    metric_card(c2, "Sensitivity", f"{sens:.3f}")
    metric_card(c3, "Specificity", f"{spec:.3f}")
    metric_card(c4, "F1 Score",    f"{f1:.3f}")
    metric_card(c5, "Brier Score", f"{brier:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y, calibrated)
        ax.plot(fpr, tpr, color="#4f8ef7", lw=2, label=f"AUC = {auroc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Calibrated Stacked Model)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown('<div class="section-header">Precision–Recall Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        precision, recall, _ = precision_recall_curve(y, calibrated)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color="#e07b39", lw=2, label=f"PR AUC = {pr_auc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred: Survive", "Pred: Death"],
                    yticklabels=["Actual: Survive", "Actual: Death"])
        ax.set_title(f"Confusion Matrix — {threshold_mode}")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r2:
        st.markdown('<div class="section-header">Calibration Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        prob_true, prob_pred = calibration_curve(y, calibrated, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", color="#2ca02c", label="Calibrated model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Threshold comparison table ─────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Threshold Strategy Comparison</div>', unsafe_allow_html=True)

    def evaluate_threshold(name, t):
        p = (calibrated >= t).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y, p).ravel()
        return {
            "Strategy": name,
            "Threshold": round(float(t), 4),
            "AUC": round(float(auroc), 4),
            "Sensitivity": round(tp_ / (tp_ + fn_), 4),
            "Specificity": round(tn_ / (tn_ + fp_), 4),
            "Precision":   round(tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0, 4),
            "F1":          round(float(f1_score(y, p)), 4),
            "TP": int(tp_), "FP": int(fp_), "FN": int(fn_), "TN": int(tn_),
        }

    comp_df = pd.DataFrame([
        evaluate_threshold("Safety (Cost 9:1)", t_cost),
        evaluate_threshold("Balanced (F1)",     t_f1),
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 — Patient Risk Prediction
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Patient Risk Prediction":
    st.title("🔬 Individual Patient Risk Prediction")

    if not data_loaded or not models_loaded:
        show_setup_warning(
            ("data file" if not data_loaded else "") +
            (", model files" if not models_loaded else "")
        )
        st.stop()

    mode = st.radio("Input mode", ["Select patient from dataset", "Enter values manually"], horizontal=True)

    if mode == "Select patient from dataset":
        patient_id = st.selectbox("Patient ID (patientunitstayid)", df["patientunitstayid"].values[:500])
        patient_row = X[df["patientunitstayid"] == patient_id]
        actual_outcome = int(df.loc[df["patientunitstayid"] == patient_id, "bad_outcome"].values[0])
    else:
        st.info("Fill in patient values below (defaults are dataset medians).")
        medians = X.median()
        col_inputs = st.columns(3)
        manual_vals = {}
        for i, feat in enumerate(features):
            col = col_inputs[i % 3]
            manual_vals[feat] = col.number_input(feat, value=float(medians[feat]), format="%.4f")
        patient_row = pd.DataFrame([manual_vals])
        actual_outcome = None

    # Threshold
    threshold_mode = st.selectbox("Threshold strategy", ["Safety (Cost 9:1)", "Balanced (F1)"], index=0)
    threshold = t_cost if threshold_mode == "Safety (Cost 9:1)" else t_f1

    if st.button("🔮 Predict Risk", type="primary"):
        cal_prob, base_preds = predict_patient(patient_row, rf, xgb, cat, lr, meta, calibrator)
        risk_class = "HIGH RISK ⚠️" if cal_prob >= threshold else "LOW RISK ✅"
        risk_html_class = "risk-high" if cal_prob >= threshold else "risk-low"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Calibrated Mortality Probability</div>
            <div class="metric-value">{cal_prob:.1%}</div>
        </div>""", unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Classification</div>
            <div class="{risk_html_class}">{risk_class}</div>
        </div>""", unsafe_allow_html=True)

        if actual_outcome is not None:
            outcome_label = "Death / Bad Outcome" if actual_outcome == 1 else "Survived"
            c3.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Actual Outcome</div>
                <div class="metric-value">{outcome_label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Base model probabilities bar chart
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-header">Base Model Predictions</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            names = list(base_preds.keys())
            vals  = list(base_preds.values())
            colors = ["#4f8ef7" if v < threshold else "#d62728" for v in vals]
            bars = ax.barh(names, vals, color=colors)
            ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.3f}")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted Probability")
            ax.set_title("Base Model Outputs")
            ax.legend()
            for bar, val in zip(bars, vals):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_r:
            st.markdown('<div class="section-header">Top SHAP Feature Drivers</div>', unsafe_allow_html=True)
            shap_values_patient = explainer.shap_values(patient_row)
            shap_series = pd.Series(shap_values_patient[0], index=features)
            top_shap = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(10)

            fig, ax = plt.subplots(figsize=(5, 3.8))
            colors_shap = ["#d62728" if v > 0 else "#2ca02c" for v in top_shap.values]
            ax.barh(top_shap.index[::-1], top_shap.values[::-1], color=colors_shap[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on model output)")
            ax.set_title("Top 10 Feature Contributions")
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 3 — Feature Importance
# ═══════════════════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.title("📈 Global Feature Importance")

    if not data_loaded or not models_loaded:
        show_setup_warning(
            ("data file" if not data_loaded else "") +
            (", model files" if not models_loaded else "")
        )
        st.stop()

    top_n = st.slider("Show top N features", 5, 30, 15)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">SHAP Summary (XGBoost)</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP values…"):
            sample_X = X.sample(min(500, len(X)), random_state=42)
            sv = explainer.shap_values(sample_X)

        fig, ax = plt.subplots(figsize=(5, max(4, top_n * 0.35)))
        mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=features).nlargest(top_n)
        mean_abs_shap = mean_abs_shap.sort_values()
        ax.barh(mean_abs_shap.index, mean_abs_shap.values, color="#4f8ef7")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Features by Mean |SHAP|")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown('<div class="section-header">XGBoost Feature Importance (gain)</div>', unsafe_allow_html=True)
        xgb_importance = pd.Series(
            xgb.feature_importances_, index=features
        ).nlargest(top_n).sort_values()

        fig, ax = plt.subplots(figsize=(5, max(4, top_n * 0.35)))
        ax.barh(xgb_importance.index, xgb_importance.values, color="#e07b39")
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title(f"Top {top_n} XGBoost Features")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # RF importance
    st.markdown('<div class="section-header">Random Forest Feature Importance</div>', unsafe_allow_html=True)
    rf_importance = pd.Series(
        rf.feature_importances_, index=features
    ).nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(rf_importance.index, rf_importance.values, color="#2ca02c")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Random Forest Features")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Feature table
    st.markdown("---")
    st.markdown('<div class="section-header">Combined Importance Table</div>', unsafe_allow_html=True)
    importance_df = pd.DataFrame({
        "Feature": features,
        "SHAP (mean |val|)":   pd.Series(np.abs(sv).mean(axis=0), index=features).values,
        "XGBoost (gain)":      xgb.feature_importances_,
        "RF (Gini)":           rf.feature_importances_,
    })
    importance_df = importance_df.sort_values("SHAP (mean |val|)", ascending=False).head(top_n)
    st.dataframe(importance_df.reset_index(drop=True), use_container_width=True, hide_index=True)
