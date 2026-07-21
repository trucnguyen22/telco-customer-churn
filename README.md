# Telco Customer Churn Analysis

An end-to-end churn analysis on the [Telco Customer Churn dataset](data/WA_Fn-UseC_-Telco-Customer-Churn.csv) (7,043 telecom customers), following a standard data-analysis workflow: data understanding → cleaning → EDA → feature engineering → modeling → evaluation → interpretation.

## Contents

- `telco_churn_analysis.ipynb` — the full notebook (7 parts, see below)
- `WA_FnUseC_TelcoCustomerChurn.csv` — the raw dataset
- `requirements.txt` — Python dependencies

## Notebook structure

| Part | Stage | What it covers |
|---|---|---|
| 1 | Data Understanding | Load raw data, inspect structure, dtypes, initial data-quality signals |
| 2 | Data Preparation | Fix `TotalCharges` dtype, handle blank rows, recode `SeniorCitizen`, sanity checks |
| 3 | Exploratory Data Analysis | Target balance, churn rate by segment, numeric distributions, tenure patterns |
| 4 | Feature Engineering | Encode categoricals, scale numeric features, train/test split |
| 5 | Model Training | Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting |
| 6 | Model Evaluation & Comparison | Accuracy, precision, recall, F1, ROC-AUC; ROC curves; confusion matrices |
| 7 | Feature Importance & Recommendations | Best-model interpretation, business recommendations, caveats, next steps |

## Key findings

- Contract length is the single strongest churn driver — month-to-month customers churn at roughly 15x the rate of two-year customers.
- Tenure, `MonthlyCharges`, and `TotalCharges` dominate feature importance; churn risk is highest in the first 12 months (~48%) and drops to ~7% by year six.
- Random Forest gives the best balanced predictive performance (F1 ≈ 0.63); all models converge around 0.82–0.85 ROC-AUC.
- Findings are correlational, not causal — see the notebook's caveats section for what would be needed to confirm causal effects (e.g. an A/B test on contract-upgrade incentives).

## Running locally

```bash
pip install -r requirements.txt
jupyter notebook telco_churn_analysis.ipynb
```
