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

Figure 1: Distribution of target variable, bad_outcome

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

---
# Exploratory Data Analysis (EDA)
Key exploratory analyses conducted include:
- Distribution of bad outcomes across the ICU population
- Feature correlation analysis and risk stratification patterns
- Distribution of ICU types across different levels of care in the dataset

## Visualizing our data: 
<img width="385" height="279" alt="Screenshot 2026-05-01 at 23 09 29" src="https://github.com/user-attachments/assets/96ddab68-f46f-4089-8d76-799393575806" />
Figure 2: Distribution of target variable, bad_outcome

<img width="1211" height="289" alt="Screenshot 2026-05-01 at 23 18 01" src="https://github.com/user-attachments/assets/ffd00147-1a4f-4007-aa07-c0d30f97b6c5" />
Figure 3: Distribution of target
variable, bad_outcome, across deaths and readmits (cleaned dataset version included)

---

<img width="743" height="580" alt="Screenshot 2026-05-01 at 23 16 42" src="https://github.com/user-attachments/assets/45a87e83-339b-4523-8bc6-ad5b28958b2e" />
Figure 3: Feature correlation matrix

---
# Our Modelling Approach
Baseline Model:
- Logistic Regression (interpretable baseline model)
- Advanced Models
- Random Forest
- XGBoost
- CatBoost
Final Model:
- Stacked ensemble model combining all base learners with a logistic regression meta-learner
Rationale for this approach
- Preserves interpretability through our linear meta-model
- Improves predictive performance through ensemble learning
- Balances model robustness for clinical contexts

---
# Model Training

Tools Used:
- Python
- Pandas / NumPy
- Scikit-learn
- RandomForest
- XGBoost
- CatBoost
- SHAP
- Streamlit
  
Training Pipeline:
- Data (Train-test) splitting: 80/20 train-test split with out-of-fold (OOF) validation
- Cross-validation: Stratified K-Fold cross-validation with OOF predictions for robust evaluation
- Hyperparameter optimization: Threshold tuning using cost-sensitive (9:1) and F1-optimized strategies
- Probability calibration: Isotonic regression applied to improve probability reliability
- Final modeling: Stacked ensemble built using our calibrated base models and meta-learner

--- 
# Results 
<img width="1422" height="736" alt="Baseline Table" src="https://github.com/user-attachments/assets/992e5822-c75a-44d3-a69e-d766301b4101" />
## Baseline Model 

<img width="1796" height="876" alt="Final Model" src="https://github.com/user-attachments/assets/6895c333-21ec-4d4f-a8d3-2ed86c585d10" />
## Final Stacked Model
---
# Model Explainability 
SHAP (SHapley Additive exPlanations) is used to explain individual model predictions.

It enables healthcare providers to see:
- Which features contribute to increased risk
- Which features contribute to reduced risk
- The key factors driving each specific prediction

<img width="713" height="205" alt="Screenshot 2026-05-01 at 23 41 53" src="https://github.com/user-attachments/assets/67e654a7-df62-41f8-b22f-c8fcebcb7b80" />

Figure 4: Global Explainability for the model

<img width="438" height="277" alt="Screenshot 2026-05-01 at 23 42 14" src="https://github.com/user-attachments/assets/40151d47-40ff-4f6c-bca8-abc9067e7ba5" />

Figure 5: Local Explainability for an individual patient

---
# Key Insights 
- The stacked ensemble approach led to more stable and consistent performance than any single model
- Calibration techniques improved the accuracy and reliability of predicted probabilities
- SHAP explanations enhanced transparency, making model outputs more interpretable and clinically meaningful
- The dual-threshold design supports flexible decision-making, balancing patient safety with overall performance objectives
---

# Conclusion

This project presents a complete machine learning pipeline for predicting ICU mortality risk, covering data preprocessing, model development, interpretability, and deployment.

_It is designed as a clinical decision support tool to assist decision-making rather than replace clinical expertise. This project is for educational and research purposes and not affiliated with or endorsed by any of the healthcare organizations referenced in our documentation._

---
# Future Work
- Prospective testing in real-world clinical environments
- Deployment integration with electronic health record (EHR) systems
- Continuous monitoring for model drift and performance changes
- Integration of healthcare provider feedback to guide system improvements
- A second-opinion toggle powered by a supporting model to provide alternative risk estimates for comparison

---
# How To Run 
Using Terminal, install dependencies as follows
```bash
# install dependencies
pip install -r requirements.txt
git clone https://github.com/your-username/DATA4382_ICURiskPrediction.git
# run streamlit app
streamlit run icu_deployment.py
```

---
# Repository Structure
- README.md — Project documentation
- icu_dashboard.py — Main Streamlit application
- requirements.txt — Project dependencies
- model .pkl files - Trained model artifacts for the stacked ensemble
- final_merged_cleaned_preprocessed.csv - Processed dataset

Modeling Pipeline: 
- Step 1: Remerging data from eICU SQL files (Remerging_data_step1.ipynb)
- Step 2: Rough logistic regression model post remerging (Rough_baseline_results_step2.ipynb)
- Step 3: Data preprocessing and cleaning (Data_cleaning_step3.ipynb)
- Step 4: Baseline logistic regression model after data preprocessing (Baseline_after_preprocessing_step4.ipynb)
- Step 5: Finding the best stacked ensemble combination strategy (Best_stackedmodel_step5.ipynb)
- Step 6: Building the Streamlit application (ICU_Deployment_Streamlit.ipynb)
- Final processed dataset after following all steps (final_merged_merged_cleaned_preprocessed.csv)

Presentations: 
- Milestone 1 presentation (ICU_DB_P1.pdf)
- Milestone 2 presentation (ICU_DB_P2.pdf)
- Final Executive Presentation (Executive ICU Presentation.pdf)


