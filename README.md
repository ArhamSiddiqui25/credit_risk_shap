# Explainable Machine Learning for Credit Risk Prediction

A comparative study of model accuracy and interpretability using SHAP (B.Tech CSE minor project).

**Live app:** https://creditriskshap-aw6dg4aewsm5mkemgvgvnn.streamlit.app

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditriskshap-aw6dg4aewsm5mkemgvgvnn.streamlit.app)

Four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) are trained on a balanced
sample of LendingClub loans to predict default, compared on the same test set, and explained with SHAP.
The best model by AUC-ROC (XGBoost) is deployed in a Streamlit app that gives a default-risk estimate
together with a SHAP explanation of that prediction.

## Try the app

1. Open the live app link above (a sleeping app may take a few seconds to wake up).
2. Pick an example from **Load an example** (lower-risk, typical or higher-risk applicant), or enter your own
   loan amount, interest rate, annual income, debt-to-income ratio and loan grade in the sidebar.
3. Read the **predicted default probability** and the **main drivers**, then look at the SHAP chart to see
   how each feature pushed the prediction up (red) or down (blue).
4. Optional inputs (loan term, purpose, home ownership, income verification, employment length) are under
   *More details*. Anything not asked for is set to the average of the training data.

## Results (balanced test set, 21,401 loans)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6692 | 0.6829 | 0.6319 | 0.6564 | 0.7340 |
| Decision Tree | 0.6615 | 0.6743 | 0.6247 | 0.6485 | 0.7254 |
| Random Forest | 0.6716 | 0.6679 | 0.6827 | 0.6752 | 0.7351 |
| XGBoost | 0.6732 | 0.6724 | 0.6756 | 0.6740 | 0.7383 |

The differences between the top three models are small (XGBoost is ahead of Logistic Regression by 0.0043
AUC) and were not tested for statistical significance. In the SHAP analysis, interest rate and the 60-month
term were the most influential features for all four models.

The first 300,000 records of the LendingClub accepted-loans file were used; 265,776 had a settled outcome
(Fully Paid or Charged Off) and were balanced to 107,002 loans (80/20 stratified split, 15 predictors,
29 model features after encoding).

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `requirements.txt` | Python packages for the app |
| `models/model.joblib` | Trained XGBoost model |
| `models/scaler.joblib` | StandardScaler fitted on the training set (9 numeric columns) |
| `models/feature_columns.joblib` | The 29 model input columns, in order |
| `README.md` | This file |

## How the app works

1. The sidebar inputs are turned into the same 29 columns the model was trained on (one-hot encoded
   categories, grade mapped A–G to 1–7, loan amount and income capped as in training).
2. The monthly instalment is calculated from the loan amount, interest rate and term.
3. The nine numeric columns are scaled with the saved scaler.
4. XGBoost gives the probability of default, and SHAP's `TreeExplainer` explains that single prediction.

## Run the app locally

```bash
git clone https://github.com/ArhamSiddiqui25/credit_risk_shap.git
cd credit_risk_shap
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 was used for the deployed app.

## Deployment

The app is hosted on Streamlit Community Cloud from the `main` branch of this repository
(main file `app.py`). Any commit to `main` redeploys it automatically.

## Data

LendingClub accepted loans, 2007–2018 Q4 (Kaggle dataset `wordsforthewise/lending-club`, file
`accepted_2007_to_2018Q4.csv.gz`). The raw data is not included in this repository because of its size.

## Notes and limitations

- The model was trained on a balanced sample, so the predicted probability is higher than real-world
  default rates and should be read as a risk score, not an exact chance.
- Interest rate and grade are highly correlated (0.97), and so are loan amount and instalment (0.95), so
  SHAP credit can be split between these related features.
- Only the first 300,000 rows of the dataset were used.
- Educational project; not for real lending decisions.

## Author

Arham Siddiqui (Enrolment No. A7605223065), B.Tech CSE.
Faculty guide: Dr. Shalini Agarwal.
