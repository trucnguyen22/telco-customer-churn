# Telco Customer Churn Analysis

An end-to-end churn analysis on the Telco Customer Churn dataset (7,043 telecom customers), following a standard data-analysis workflow. The goal is to identify which features most strongly drive customer churn — for example, how long a customer stays, whether support services matter, and how much monthly charges affect the outcome — then build models that test these findings and predict churn reliably on new, unseen data.

## Notebook structure

| Part | Stage | What it covers |
|---|---|---|
| 1 | Setup & Problem Framing | Objective, dataset, imports, conventions |
| 2 | Data Loading & Understanding | Load raw data; inspect shape, dtypes, and data-quality signals |
| 3 | Data Cleaning | Fix `TotalCharges` dtype, drop blank rows, recode `SeniorCitizen`, sanity checks |
| 4 | Exploratory Data Analysis | Target balance, churn rate by segment, distributions, tenure curve, correlation check |
| 5 | Feature Engineering | Create features (`num_addon_services`, `charges_delta`), encoding strategy, feature selection |
| 6 | Modeling | Train/test split, preprocessing pipeline (encode + scale), train 5 models, cross-validation |
| 7 | Evaluation & Comparison | Accuracy, precision, recall, F1, ROC-AUC; ROC curves; confusion matrices |
| 8 | Interpretation & Recommendations | Feature importance, odds ratios, business recommendations, caveats, next steps |

## Key findings

### Dataset features & churn drivers

- **Churn is moderately imbalanced (~27%).** About one in four customers left. Because always predicting "no churn" would already look ~73% accurate, the analysis judges models on precision, recall, F1, and ROC-AUC rather than accuracy alone.
- **Contract type is the strongest and most consistent driver.** Month-to-month customers churn at roughly 15 times the rate of two-year customers, and this shows up in every view — segment rates, model feature importance, and logistic-regression coefficients. Encouraging longer contracts is the clearest lever for retention.
- **The first year is the highest-risk period.** Roughly 48% of customers churn within their first 12 months, falling steadily to around 7% by year six. New customers are the priority for early retention outreach.
- **Certain service and payment types flag higher risk.** Fiber-optic internet (~42%) and electronic-check payment (~45%) churn well above the ~27% average, while customers with online-security or tech-support add-ons churn less. These are specific groups worth investigating and targeting.
- **Tenure and charges predict well, but read them carefully.** Tenure and monthly charges rank among the top predictors, and an engineered feature — how a customer's current rate compares to their lifetime average (`charges_delta`) — also earned a place in the tree models. Total charges was dropped as redundant (it is almost exactly tenure × monthly charges). Monthly charges still changes direction between the single-variable view and the combined model because it is entangled with fiber and streaming, so the contract, service, and payment signals stay the most dependable.

### Model comparison & selection

- **All five models perform about the same.** Logistic Regression, KNN, Decision Tree, Random Forest, and Gradient Boosting all land around 0.81–0.83 ROC-AUC, in both cross-validation and on the held-out test set. No model clearly wins on ranking quality.
- **Because the models tie, selection comes down to the error trade-off.** Random Forest gives the best overall balance (F1 ≈ 0.62), the Decision Tree catches the most churners (highest recall), and KNN is the most precise but misses the most. With every model class-balanced except KNN, their precision/recall profiles are otherwise similar.
- **Recommended default: Random Forest**, with the decision threshold tuned to the business's needs. The final choice should weigh the cost of missing a churner against the cost of contacting one who would have stayed — a business input, not a purely statistical one.
- **Caveats.** Metrics use the default 0.5 probability cutoff, which has not been tuned. KNN is the only model without class balancing, which explains its more conservative (higher-precision, lower-recall) profile. All results are correlational — confirming a cause (e.g. that contract upgrades reduce churn) would require a controlled experiment such as an A/B test. See the notebook's caveats section for details.

