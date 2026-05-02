# Machine Learning–Driven ICU Risk Prediction

# Our Motivation
Intensive Care Units (ICUs) operate in fast-paced, high-stakes environments where timely detection of patient deterioration can strongly influence survival and recovery. When high-risk patients are not identified early, it can result in avoidable complications, higher mortality rates, and suboptimal use of limited clinical resources.

This project aims to assist clinicians by generating real-time, interpretable risk estimates for patient mortality using structured electronic health record (EHR) data.

Rather than replacing clinical expertise, the goal is to complement it by offering transparent, data-driven decision support that can improve situational awareness and support more informed clinical decisions.

---
# Project Overview
This project is designed to be deployed clinical decision support tool designed to estimate ICU mortality risk using structured electronic health record (EHR) data from the eICU Collaborative Research Database.

It leverages a stacked ensemble machine learning framework combined with probability calibration to generate more reliable and clinically meaningful risk predictions.

Key components include:

- A dual-model prediction framework (one prioritizing safety and another optimized for balanced performance)
- Built-in SHAP-based explainability to interpret individual patient-level predictions
- An interactive Streamlit dashboard for clinical exploration and use
- Patient cohort-level evaluation section for assessing overall model performance, assisting data analysis and model retraining
  
- ICU Dashboard: https://data4382icuriskprediction.streamlit.app/
---

# Our Pipeline 
<img width="621" height="1834" alt="image" src="https://github.com/user-attachments/assets/01ffa0b4-16b4-48f2-96cd-5009e4062f4c" />

---
# Data
- eICU Collaborative Research Database: https://physionet.org/content/eicu-crd-demo/2.0.1/
- Dataset Information
Number of patients: 2,520
Number of features: 87
Target variable: bad_outcome (We defined this as binary mortality and readmission risk indicator)

---
# Data Preprocessing
The data was preprocessed through the following steps:
- Extracted structured data from eICU tables using SQLite
- Merged multiple tables into patient-level records
- Handled missing values and applied feature imputation
- Performed feature engineering, including clinical ratios and aggregated vital sign statistics, which allow the model to gain more insight for each patient's physiology
- Applied one-hot encoding to categorical variables
- Constructed the final modeling dataset for reproducible machine learning workflows
