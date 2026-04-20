"""
ICU Mortality Prediction Dashboard
Stacked Ensemble: LR + XGBoost + CatBoost + RF → Logistic Meta-Learner
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report, f1_score
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Mortality Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #5a7a9c;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8eef4 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #2563eb;
        margin: 4px 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe0e0 100%);
        border-left: 4px solid #dc2626;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff9f0 0%, #fff0d0 100%);
        border-left: 4px solid #d97706;
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fff4 0%, #d0f0d8 100%);
        border-left: 4px solid #16a34a;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stMetric > div { padding: 0; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "final_merged_cleaned_preprocessed.csv"
TARGET = "bad_outcome"
ID_COL = "patientunitstayid"
APACHE_FEATURES = ["acutephysiologyscore", "apachescore"]

# Thresholds from notebook
T_COST = 0.099091   # Safety / Cost 9:1
T_F1   = 0.297071   # Balanced / F1-optimal


@st.cache_data(show_spinner="Loading patient data…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource(show_spinner="Training stacked ensemble (this runs once)…")
def train_models(df):
    drop_cols = [ID_COL, TARGET]
    X = df.drop(columns=drop_cols)
    y = df[TARGET]

    pos_weight = (y == 0).sum() / (y == 1).sum()

    def get_oof(model, Xd, yd, n_splits=5):
        oof = np.zeros(len(Xd))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, va in skf.split(Xd, yd):
            model.fit(Xd.iloc[tr], yd.iloc[tr])
            oof[va] = model.predict_proba(Xd.iloc[va])[:, 1]
        return oof

    rf  = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        scale_pos_weight=pos_weight, random_state=42, verbosity=0)
    cat = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                             loss_function="Logloss", verbose=0, random_seed=42)
    lr  = LogisticRegression(max_iter=3000, class_weight="balanced")

    rf_oof  = get_oof(rf,  X, y)
    xgb_oof = get_oof(xgb, X, y)
    cat_oof = get_oof(cat, X, y)
    lr_oof  = get_oof(lr,  X, y)

    meta_X = pd.DataFrame({"rf": rf_oof, "xgb": xgb_oof, "cat": cat_oof, "lr": lr_oof})
    meta   = LogisticRegression(max_iter=3000)

    meta_oof = np.zeros(len(meta_X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(meta_X, y):
        meta.fit(meta_X.iloc[tr], y.iloc[tr])
        meta_oof[va] = meta.predict_proba(meta_X.iloc[va])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(meta_oof, y)
    calibrated = iso.transform(meta_oof)

    # Fit final models on all data for inference
    rf.fit(X, y); xgb.fit(X, y); cat.fit(X, y); lr.fit(X, y)
    meta.fit(meta_X, y)

    # SHAP explainer on XGBoost
    xgb_shap = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              scale_pos_weight=pos_weight, random_state=42, verbosity=0)
    xgb_shap.fit(X, y)
    explainer = shap.TreeExplainer(xgb_shap)

    return {
        "rf": rf, "xgb": xgb, "cat": cat, "lr": lr,
        "meta": meta, "iso": iso, "explainer": explainer,
        "X": X, "y": y,
        "meta_oof": meta_oof,
        "calibrated": calibrated,
        "feature_cols": X.columns.tolist(),
    }


def predict_patient(models, patient_row):
    """Return calibrated risk probability for a single patient."""
    X_row = patient_row[models["feature_cols"]]
    rf_p  = models["rf"].predict_proba(X_row)[:, 1]
    xgb_p = models["xgb"].predict_proba(X_row)[:, 1]
    cat_p = models["cat"].predict_proba(X_row)[:, 1]
    lr_p  = models["lr"].predict_proba(X_row)[:, 1]
    meta_in = pd.DataFrame({"rf": rf_p, "xgb": xgb_p, "cat": cat_p, "lr": lr_p})
    raw_p = models["meta"].predict_proba(meta_in)[:, 1]
    cal_p = models["iso"].transform(raw_p)
    return float(cal_p[0])


def risk_label(prob, threshold):
    if prob >= threshold * 2:
        return "HIGH", "risk-high", "🔴"
    elif prob >= threshold:
        return "ELEVATED", "risk-medium", "🟡"
    else:
        return "LOW", "risk-low", "🟢"


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────
df = load_data()
models = train_models(df)
X_all = models["X"]
y_all = models["y"]
calibrated = models["calibrated"]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ICU Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔍 Patient Explorer", "🎯 Live Predictor", "📊 Model Performance", "🌍 Cohort Overview"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Decision Threshold**")
    threshold_mode = st.selectbox(
        "Mode",
        ["Safety (Cost 9:1)  — t=0.099", "Balanced (F1-optimal)  — t=0.297"],
        help="Safety mode maximises sensitivity (catches more deaths). Balanced mode optimises F1.",
    )
    threshold = T_COST if "Safety" in threshold_mode else T_F1

    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} patients · {y_all.sum():,} bad outcomes ({y_all.mean()*100:.1f}%)")
    st.caption("Model: LR + XGB + CAT + RF → Logistic Meta + Isotonic Calibration")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Cohort Overview
# ──────────────────────────────────────────────────────────────────────────────
if page == "🌍 Cohort Overview":
    st.markdown('<p class="main-header">🌍 Cohort Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Population-level statistics across all ICU patients</p>', unsafe_allow_html=True)

    # Top KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients", f"{len(df):,}")
    c2.metric("ICU Deaths / Bad Outcomes", f"{y_all.sum():,}")
    c3.metric("Mortality Rate", f"{y_all.mean()*100:.1f}%")
    c4.metric("Median Age", f"{df['age'].median():.0f} yrs")
    c5.metric("Model AUC", "0.832")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Age Distribution by Outcome</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        survived = df[df[TARGET] == 0]["age"].dropna()
        died     = df[df[TARGET] == 1]["age"].dropna()
        ax.hist(survived, bins=25, alpha=0.6, color="#2563eb", label="Survived")
        ax.hist(died,     bins=25, alpha=0.7, color="#dc2626", label="Bad Outcome")
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<p class="section-title">APACHE Score Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[df[TARGET]==0]["apachescore"].dropna(), bins=25, alpha=0.6, color="#2563eb", label="Survived")
        ax.hist(df[df[TARGET]==1]["apachescore"].dropna(), bins=25, alpha=0.7, color="#dc2626", label="Bad Outcome")
        ax.set_xlabel("APACHE Score"); ax.set_ylabel("Count")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-title">Unit Type Breakdown</p>', unsafe_allow_html=True)
        unit_cols = {
            "unittype_CSICU": "CSICU", "unittype_CTICU": "CTICU",
            "unittype_Cardiac ICU": "Cardiac ICU", "unittype_MICU": "MICU",
            "unittype_Med-Surg ICU": "Med-Surg ICU", "unittype_Neuro ICU": "Neuro ICU",
            "unittype_SICU": "SICU",
        }
        present = {v: df[k].sum() for k, v in unit_cols.items() if k in df.columns}
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(list(present.keys()), list(present.values()), color="#3b82f6")
        ax.set_xlabel("Patient Count"); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<p class="section-title">Top Comorbidities Prevalence</p>', unsafe_allow_html=True)
        comorbids = {
            "Diabetes": df["diabetes"].sum(),
            "Cardio Hx": df["hx_cardio"].sum(),
            "Respiratory Hx": df["hx_respiratory"].sum(),
            "Renal Hx": df["hx_renal"].sum(),
            "Cancer Hx": df["hx_cancer"].sum(),
            "Immunosuppression": df["immunosuppression"].sum(),
            "Cirrhosis": df["cirrhosis"].sum(),
            "Dialysis": df["dialysis"].sum(),
        }
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(list(comorbids.keys()), list(comorbids.values()), color="#8b5cf6")
        ax.set_xlabel("# Patients"); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Risk distribution from calibrated probs
    st.markdown('<p class="section-title">Calibrated Risk Score Distribution</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(calibrated[y_all==0], bins=50, alpha=0.6, color="#2563eb", label="Survived", density=True)
    ax.hist(calibrated[y_all==1], bins=50, alpha=0.7, color="#dc2626", label="Bad Outcome", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel("Calibrated Risk Probability"); ax.set_ylabel("Density")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Patient Explorer
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Patient Explorer":
    st.markdown('<p class="main-header">🔍 Patient Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Browse the patient cohort and view individual records</p>', unsafe_allow_html=True)

    col_search, col_filter = st.columns([2, 2])
    with col_search:
        pid_input = st.text_input("Search by Patient Unit Stay ID", placeholder="e.g. 141764")
    with col_filter:
        outcome_filter = st.selectbox("Filter by Outcome", ["All", "Bad Outcome (1)", "Survived (0)"])

    filtered = df.copy()
    if pid_input.strip():
        try:
            filtered = filtered[filtered[ID_COL] == int(pid_input.strip())]
        except ValueError:
            st.warning("Enter a valid numeric ID.")
    if outcome_filter == "Bad Outcome (1)":
        filtered = filtered[filtered[TARGET] == 1]
    elif outcome_filter == "Survived (0)":
        filtered = filtered[filtered[TARGET] == 0]

    # Attach calibrated risk from OOF
    risk_series = pd.Series(calibrated, index=df.index, name="Calibrated Risk")
    filtered = filtered.join(risk_series)
    filtered["Risk Label"] = filtered["Calibrated Risk"].apply(
        lambda p: risk_label(p, threshold)[0]
    )

    display_cols = [ID_COL, "age", "apachescore", "acutephysiologyscore",
                    "Calibrated Risk", "Risk Label", TARGET]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].style
            .format({"Calibrated Risk": "{:.3f}"})
            .background_gradient(subset=["Calibrated Risk"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=320,
    )
    st.caption(f"Showing {len(filtered):,} patients")

    # Detail card
    st.markdown("---")
    st.markdown('<p class="section-title">Patient Detail Card</p>', unsafe_allow_html=True)
    if len(filtered) > 0:
        selected_id = st.selectbox(
            "Select a patient to inspect",
            options=filtered[ID_COL].tolist(),
            format_func=lambda x: f"ID {x}"
        )
        row = df[df[ID_COL] == selected_id].iloc[[0]]
        risk_prob = predict_patient(models, row)
        rl, rc, ri = risk_label(risk_prob, threshold)

        dc1, dc2, dc3 = st.columns([1, 1, 1])

        with dc1:
            st.markdown(f'<div class="metric-card {rc}"><b>{ri} Risk Level: {rl}</b><br>'
                        f'Calibrated Risk: <b>{risk_prob:.1%}</b><br>'
                        f'Threshold: {threshold:.3f}<br>'
                        f'Actual Outcome: {"⚠️ Bad Outcome" if row[TARGET].values[0]==1 else "✅ Survived"}</div>',
                        unsafe_allow_html=True)

        with dc2:
            st.markdown("**Demographics & Vitals**")
            vitals = {
                "Age": f"{row['age'].values[0]:.0f} yrs",
                "Admission Height": f"{row['admissionheight'].values[0]:.1f} cm",
                "Admission Weight": f"{row['admissionweight'].values[0]:.1f} kg",
                "BMI": f"{row['bmi'].values[0]:.1f}",
                "Heart Rate (mean)": f"{row['heartrate'].values[0]:.0f} bpm",
                "Mean BP": f"{row['meanbp'].values[0]:.0f} mmHg",
                "Resp Rate": f"{row['respiratoryrate'].values[0]:.0f} /min",
            }
            for k, v in vitals.items():
                st.write(f"**{k}:** {v}")

        with dc3:
            st.markdown("**Severity & Labs**")
            labs = {
                "APACHE Score": row["apachescore"].values[0],
                "Acute Physiology Score": row["acutephysiologyscore"].values[0],
                "Creatinine": row["creatinine"].values[0],
                "Glucose": row["glucose"].values[0],
                "Albumin": row["albumin"].values[0],
                "WBC": row["wbc"].values[0],
                "pH": row["ph"].values[0],
            }
            for k, v in labs.items():
                st.write(f"**{k}:** {v:.2f}" if isinstance(v, float) else f"**{k}:** {v}")

        # SHAP waterfall for this patient
        st.markdown("**Top Feature Drivers (SHAP)**")
        x_pat = row[models["feature_cols"]]
        shap_vals = models["explainer"].shap_values(x_pat)
        base_val = models["explainer"].expected_value

        top_shap = pd.Series(shap_vals[0], index=models["feature_cols"])
        top_shap = top_shap.reindex(top_shap.abs().sort_values(ascending=False).index).head(10)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#dc2626" if v > 0 else "#2563eb" for v in top_shap.values]
        ax.barh(top_shap.index[::-1], top_shap.values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP Value (impact on log-odds)")
        ax.set_title(f"SHAP — Patient {selected_id}")
        ax.spines[["top","right"]].set_visible(False)
        red_patch = mpatches.Patch(color="#dc2626", label="Increases risk")
        blue_patch = mpatches.Patch(color="#2563eb", label="Decreases risk")
        ax.legend(handles=[red_patch, blue_patch], fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Live Predictor
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Live Predictor":
    st.markdown('<p class="main-header">🎯 Live Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter patient data manually and get an instant ICU mortality risk estimate</p>', unsafe_allow_html=True)

    st.info("Fill in the patient data below. Missing values will be imputed with dataset medians.")

    with st.form("predictor_form"):
        st.markdown("#### 🧍 Demographics")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age    = r1c1.number_input("Age", 18.0, 110.0, 65.0, 1.0)
        height = r1c2.number_input("Height (cm)", 100.0, 220.0, 170.0, 1.0)
        weight = r1c3.number_input("Weight (kg)", 30.0, 250.0, 75.0, 1.0)
        gender = r1c4.selectbox("Gender", ["Male", "Female", "Unknown"])

        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Hispanic",
                                                "Asian", "Native American", "Unknown"])
        unit_type  = st.selectbox("ICU Unit Type", ["Med-Surg ICU", "MICU", "SICU", "CTICU",
                                                     "Neuro ICU", "CSICU", "Cardiac ICU"])
        region     = st.selectbox("Region", ["South", "Northeast", "Midwest", "West", "Unknown"])
        bed_cat    = st.selectbox("Hospital Bed Category", ["<100", "100-249", "250-499", ">=500"])

        st.markdown("#### 💊 Severity Scores")
        sc1, sc2 = st.columns(2)
        apache_score = sc1.number_input("APACHE Score",     0.0, 200.0, 35.0, 1.0)
        aps_score    = sc2.number_input("Acute Physiology Score", 0.0, 200.0, 25.0, 1.0)

        st.markdown("#### 🩸 Lab Values")
        l1, l2, l3, l4 = st.columns(4)
        creatinine = l1.number_input("Creatinine",  0.0, 30.0, 1.0, 0.1)
        glucose    = l2.number_input("Glucose",     0.0, 1000.0, 120.0, 5.0)
        albumin    = l3.number_input("Albumin",     0.0, 10.0, 3.5, 0.1)
        wbc        = l4.number_input("WBC",         0.0, 100.0, 9.0, 0.5)

        l5, l6, l7, l8 = st.columns(4)
        sodium     = l5.number_input("Sodium",      100.0, 200.0, 138.0, 1.0)
        ph         = l6.number_input("pH",          6.5, 8.0, 7.4, 0.01)
        bun        = l7.number_input("BUN",         0.0, 200.0, 18.0, 1.0)
        bilirubin  = l8.number_input("Bilirubin",   0.0, 50.0, 0.8, 0.1)

        st.markdown("#### ❤️ Vitals")
        v1, v2, v3, v4 = st.columns(4)
        heart_rate = v1.number_input("Heart Rate (bpm)", 20.0, 250.0, 80.0, 1.0)
        mean_bp    = v2.number_input("Mean BP (mmHg)",   20.0, 200.0, 80.0, 1.0)
        resp_rate  = v3.number_input("Resp Rate (/min)", 4.0, 60.0, 18.0, 1.0)
        sao2_min   = v4.number_input("Min SpO2 (%)",     50.0, 100.0, 95.0, 0.5)

        st.markdown("#### 🏥 Comorbidities")
        cm1, cm2, cm3, cm4 = st.columns(4)
        diabetes   = cm1.checkbox("Diabetes")
        hx_cardio  = cm2.checkbox("Cardiovascular Hx")
        hx_resp    = cm3.checkbox("Respiratory Hx")
        hx_renal   = cm4.checkbox("Renal Hx")
        cm5, cm6, cm7, cm8 = st.columns(4)
        hx_cancer  = cm5.checkbox("Cancer Hx")
        immunosupp = cm6.checkbox("Immunosuppression")
        cirrhosis  = cm7.checkbox("Cirrhosis")
        dialysis   = cm8.checkbox("Dialysis")

        submitted = st.form_submit_button("🔮 Predict ICU Mortality Risk", use_container_width=True)

    if submitted:
        # Build row matching feature_cols
        medians = X_all.median()
        row_dict = medians.to_dict()

        # Overwrite with user inputs
        row_dict.update({
            "age": age,
            "admissionheight": height,
            "admissionweight": weight,
            "bmi": weight / ((height / 100) ** 2),
            "apachescore": apache_score,
            "acutephysiologyscore": aps_score,
            "creatinine": creatinine,
            "glucose": glucose,
            "albumin": albumin,
            "wbc": wbc,
            "sodium": sodium,
            "ph": ph,
            "bun": bun,
            "bilirubin": bilirubin,
            "heartrate": heart_rate,
            "meanbp": mean_bp,
            "respiratoryrate": resp_rate,
            "sao2_min": sao2_min,
            "diabetes": int(diabetes),
            "hx_cardio": int(hx_cardio),
            "hx_respiratory": int(hx_resp),
            "hx_renal": int(hx_renal),
            "hx_cancer": int(hx_cancer),
            "immunosuppression": int(immunosupp),
            "cirrhosis": int(cirrhosis),
            "dialysis": int(dialysis),
            # Gender dummies
            "gender_Male":    int(gender == "Male"),
            "gender_Unknown": int(gender == "Unknown"),
            # Ethnicity dummies
            "ethnicity_Caucasian":        int(ethnicity == "Caucasian"),
            "ethnicity_Hispanic":         int(ethnicity == "Hispanic"),
            "ethnicity_Asian":            int(ethnicity == "Asian"),
            "ethnicity_African American": int(ethnicity == "African American"),
            "ethnicity_Native American":  int(ethnicity == "Native American"),
            "ethnicity_Unknown":          int(ethnicity == "Unknown"),
            # Unit type dummies
            "unittype_CSICU":        int(unit_type == "CSICU"),
            "unittype_CTICU":        int(unit_type == "CTICU"),
            "unittype_Cardiac ICU":  int(unit_type == "Cardiac ICU"),
            "unittype_MICU":         int(unit_type == "MICU"),
            "unittype_Med-Surg ICU": int(unit_type == "Med-Surg ICU"),
            "unittype_Neuro ICU":    int(unit_type == "Neuro ICU"),
            "unittype_SICU":         int(unit_type == "SICU"),
            # Region dummies
            "region_Northeast": int(region == "Northeast"),
            "region_South":     int(region == "South"),
            "region_West":      int(region == "West"),
            "region_Unknown":   int(region == "Unknown"),
        })

        row_df = pd.DataFrame([row_dict])[models["feature_cols"]]
        prob = predict_patient(models, row_df)
        rl, rc, ri = risk_label(prob, threshold)

        st.markdown("---")
        st.markdown("### Prediction Result")
        pr1, pr2, pr3 = st.columns([1, 1, 2])

        with pr1:
            st.markdown(
                f'<div class="metric-card {rc}" style="text-align:center; padding:24px">'
                f'<div style="font-size:3rem">{ri}</div>'
                f'<div style="font-size:1.8rem; font-weight:700">{prob:.1%}</div>'
                f'<div style="font-size:1rem">ICU Mortality Risk</div>'
                f'<div style="margin-top:8px; font-weight:600">Risk Level: {rl}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with pr2:
            st.markdown("**Interpretation**")
            if rl == "HIGH":
                st.error(f"⚠️ **High risk** of bad outcome.\nProbability {prob:.1%} exceeds {threshold*2:.1%} (2× threshold).\nConsider urgent clinical review.")
            elif rl == "ELEVATED":
                st.warning(f"⚡ **Elevated risk** detected.\nProbability {prob:.1%} exceeds threshold {threshold:.3f}.\nClose monitoring recommended.")
            else:
                st.success(f"✅ **Low risk** of bad outcome.\nProbability {prob:.1%} is below threshold {threshold:.3f}.")

            st.caption(f"Threshold mode: {'Safety (9:1 cost)' if threshold==T_COST else 'Balanced (F1-optimal)'}")

        with pr3:
            # Mini gauge
            fig, ax = plt.subplots(figsize=(5, 2.5))
            risk_pct = min(prob, 1.0)
            bar_color = "#dc2626" if rl == "HIGH" else ("#d97706" if rl == "ELEVATED" else "#16a34a")
            ax.barh(["Risk"], [risk_pct], color=bar_color, height=0.4)
            ax.barh(["Risk"], [1 - risk_pct], left=[risk_pct], color="#e2e8f0", height=0.4)
            ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold {threshold:.3f}")
            ax.set_xlim(0, 1); ax.set_xlabel("Calibrated Probability")
            ax.set_title("Risk Gauge"); ax.legend(fontsize=8)
            ax.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # SHAP for this input
        st.markdown("**Feature Impact (SHAP)**")
        shap_vals = models["explainer"].shap_values(row_df)
        top_shap = pd.Series(shap_vals[0], index=models["feature_cols"])
        top_shap = top_shap.reindex(top_shap.abs().sort_values(ascending=False).index).head(12)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#dc2626" if v > 0 else "#2563eb" for v in top_shap.values]
        ax.barh(top_shap.index[::-1], top_shap.values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title("Top Features Driving This Prediction")
        ax.spines[["top","right"]].set_visible(False)
        red_patch = mpatches.Patch(color="#dc2626", label="Increases risk")
        blue_patch = mpatches.Patch(color="#2563eb", label="Decreases risk")
        ax.legend(handles=[red_patch, blue_patch])
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Out-of-fold evaluation metrics for the stacked ensemble</p>', unsafe_allow_html=True)

    probs = calibrated

    # Top metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    y_pred_thresh = (probs >= threshold).astype(int)
    auc_val   = roc_auc_score(y_all, probs)
    acc_val   = accuracy_score(y_all, y_pred_thresh)
    prec_val  = precision_score(y_all, y_pred_thresh, zero_division=0)
    rec_val   = recall_score(y_all, y_pred_thresh, zero_division=0)
    f1_val    = f1_score(y_all, y_pred_thresh)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC",   f"{auc_val:.3f}")
    m2.metric("Accuracy",  f"{acc_val:.3f}")
    m3.metric("Precision", f"{prec_val:.3f}")
    m4.metric("Recall",    f"{rec_val:.3f}")
    m5.metric("F1-Score",  f"{f1_val:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_all, y_pred_thresh)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"], ax=ax)
        ax.set_title(f"Threshold = {threshold:.3f}")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
        st.dataframe(
            pd.DataFrame({
                "Metric": ["Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV"],
                "Value":  [f"{sens:.3f}", f"{spec:.3f}", f"{ppv:.3f}", f"{npv:.3f}"]
            }).set_index("Metric"),
            use_container_width=True
        )

    with col2:
        st.markdown('<p class="section-title">ROC Curve</p>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_all, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-title">Precision-Recall Curve</p>', unsafe_allow_html=True)
        precision, recall, _ = precision_recall_curve(y_all, probs)
        pr_auc = auc(recall, precision)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(recall, precision, color="#7c3aed", lw=2, label=f"PR AUC = {pr_auc:.3f}")
        ax.axhline(y_all.mean(), color="gray", linestyle="--", label="Baseline")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<p class="section-title">Global SHAP Feature Importance</p>', unsafe_allow_html=True)
        shap_vals_all = models["explainer"].shap_values(X_all)
        mean_abs_shap = pd.Series(
            np.abs(shap_vals_all).mean(axis=0),
            index=models["feature_cols"]
        ).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.barh(mean_abs_shap.index[::-1], mean_abs_shap.values[::-1], color="#f59e0b")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 15 Global Features")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Threshold comparison table
    st.markdown('<p class="section-title">Threshold Strategy Comparison</p>', unsafe_allow_html=True)
    rows = []
    for name, t in [("Safety (Cost 9:1)", T_COST), ("Balanced (F1-optimal)", T_F1), ("Default (0.5)", 0.5)]:
        preds = (probs >= t).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_all, preds).ravel()
        rows.append({
            "Strategy": name,
            "Threshold": f"{t:.3f}",
            "AUC": f"{auc_val:.3f}",
            "Sensitivity": f"{tp2/(tp2+fn2):.3f}",
            "Specificity": f"{tn2/(tn2+fp2):.3f}",
            "Precision": f"{tp2/(tp2+fp2) if (tp2+fp2)>0 else 0:.3f}",
            "F1": f"{f1_score(y_all, preds):.3f}",
            "TP": tp2, "FP": fp2, "FN": fn2, "TN": tn2,
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

    # Model comparison table from notebook
    st.markdown('<p class="section-title">Model Comparison (from training)</p>', unsafe_allow_html=True)
    comp_df = pd.DataFrame([
        {"Model": "SEM Strategy (Paper-style)", "Cal. AUC": 0.8237, "Accuracy": 0.84, "Precision (1)": 0.90, "Recall (1)": 0.33, "F1 (1)": 0.48},
        {"Model": "XGBoost + CatBoost",          "Cal. AUC": 0.8257, "Accuracy": 0.84, "Precision (1)": 0.87, "Recall (1)": 0.36, "F1 (1)": 0.51},
        {"Model": "✅ LR+XGB+CAT+RF → Meta (best)", "Cal. AUC": 0.8318, "Accuracy": 0.84, "Precision (1)": 0.80, "Recall (1)": 0.41, "F1 (1)": 0.54},
    ]).set_index("Model")
    st.dataframe(comp_df, use_container_width=True)
