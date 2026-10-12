### \# Telecom Churn Predictor

### 

Predicting which telecom customers will leave, using the IBM Telco dataset (7,043 customers, 26.5% churn).

\## Key findings

\- \*\*Contract type is the strongest signal:\*\* month-to-month customers churn at 42.7% vs 2.8% on two-year contracts.

\- \*\*The first 6 months are the danger zone:\*\* 52.9% churn, falling to 9.5% after 4 years.

\- \*\*Electronic check payers churn at 45%\*\*, about 3x customers on automatic payments.

\- \*\*Fiber optic (41.9%) and no tech support (41.6%)\*\* point to premium customers who don't get help when things go wrong.

\## Model

| Model | CV ROC AUC | Churner recall |
|---|---|---|
| Logistic regression (balanced) | 0.846 ± 0.012 | 78.3% |
| Gradient boosting (balanced) | 0.845 ± 0.011 | 79.7% |

Both models tied, so I chose \*\*logistic regression\*\*: it's simpler and its coefficients are easy to explain.

Class weighting raised churner recall from 56% to 78%, because missing a churner costs more than a wasted retention offer.

\## Data cleaning

`TotalCharges` had 11 blank values. All were customers with tenure 0 (not yet billed), so they were set to 0 rather than the mean.

\## Run it

&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install pandas scikit-learn jupyter

&#x20;   jupyter lab   # open churn.ipynb


