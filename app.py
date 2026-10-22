"""
Explainable Credit Risk Prediction - Streamlit app
==================================================
Minor Project: Explainable Machine Learning for Credit Risk Prediction:
A Comparative Study of Model Accuracy and Interpretability Using SHAP

The app loads the XGBoost model, the fitted StandardScaler and the ordered
feature list that were exported from the training notebook, turns a few
applicant details into the 29 model inputs, and shows:
  * the predicted probability of default, and
  * a SHAP waterfall plot that explains that prediction.

Run locally with:   streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

# --- settings that must match the training notebook ------------------------------------
NUM_COLS = ["loan_amnt", "int_rate", "annual_inc", "dti", "installment",
            "open_acc", "revol_bal", "revol_util", "total_acc"]        # scaled columns
GRADE_MAP = {g: i for i, g in enumerate("ABCDEFG", 1)}               # A=1 ... G=7
CAPS = {"annual_inc": (18000.0, 260000.0),                           # 1st-99th percentile caps
        "loan_amnt": (1800.0, 35000.0)}                              # used during training
THRESHOLD = 0.5

# Baseline category of each one-hot group was dropped in training (drop_first=True),
# so choosing the baseline simply leaves all of its dummy columns at 0.
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "medical", "moving", "vacation", "house", "wedding",
            "renewable_energy", "other", "car"]
HOME = ["MORTGAGE", "RENT", "OWN", "ANY"]
VERIFICATION = ["Source Verified", "Verified", "Not Verified"]
TERMS = [36, 60]
EMP_LENGTHS = {"Less than 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
               "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
               "10 or more years": 10}

NICE = {"loan_amnt": "Loan amount", "int_rate": "Interest rate", "annual_inc": "Annual income",
        "dti": "Debt-to-income ratio", "installment": "Monthly instalment",
        "open_acc": "Open accounts", "revol_bal": "Revolving balance",
        "revol_util": "Revolving utilisation", "total_acc": "Total accounts",
        "grade": "Loan grade", "emp_length": "Employment length",
        "term_ 60 months": "60-month term",
        "home_ownership_MORTGAGE": "Has a mortgage", "home_ownership_OWN": "Owns home",
        "home_ownership_RENT": "Rents home",
        "verification_status_Verified": "Income verified",
        "verification_status_Source Verified": "Income source verified"}


def nice(name):
    if name in NICE:
        return NICE[name]
    if name.startswith("purpose_"):
        return "Purpose: " + name[len("purpose_"):].replace("_", " ")
    return name


# --- core logic (no Streamlit calls, so it can be tested on its own) -------------------
def monthly_instalment(amount, rate_pct, months):
    """Standard loan instalment formula, used because the app does not ask for it."""
    r = rate_pct / 1200.0
    if r == 0:
        return amount / months
    return amount * r / (1 - (1 + r) ** (-months))


def build_raw_row(columns, scaler, loan_amnt, int_rate, annual_inc, dti, grade,
                  term=36, purpose="debt_consolidation", home="MORTGAGE",
                  verification="Source Verified", emp_length=10,
                  open_acc=None, revol_bal=None, revol_util=None, total_acc=None):
    """Return a one-row DataFrame with the 29 model columns, in raw (unscaled) units."""
    means = dict(zip(scaler.feature_names_in_, scaler.mean_))        # average borrower
    loan_amnt = float(np.clip(loan_amnt, *CAPS["loan_amnt"]))        # same caps as training
    annual_inc = float(np.clip(annual_inc, *CAPS["annual_inc"]))
    row = pd.Series(0.0, index=columns)
    row["loan_amnt"] = loan_amnt
    row["int_rate"] = int_rate
    row["annual_inc"] = annual_inc
    row["dti"] = dti
    row["installment"] = monthly_instalment(loan_amnt, int_rate, term)
    row["open_acc"] = means["open_acc"] if open_acc is None else open_acc
    row["revol_bal"] = means["revol_bal"] if revol_bal is None else revol_bal
    row["revol_util"] = means["revol_util"] if revol_util is None else revol_util
    row["total_acc"] = means["total_acc"] if total_acc is None else total_acc
    row["grade"] = GRADE_MAP[grade]
    row["emp_length"] = emp_length
    for col in (f"purpose_{purpose}", f"home_ownership_{home}",
                f"verification_status_{verification}", f"term_ {term} months"):
        if col in row.index:                                         # baseline has no column
            row[col] = 1.0
    return row.to_frame().T.astype(float)


def scale_row(raw, scaler):
    scaled = raw.copy()
    scaled[NUM_COLS] = scaler.transform(raw[NUM_COLS])
    return scaled


def predict_default(model, scaled):
    return float(model.predict_proba(scaled)[:, 1][0])


def explain(explainer, scaled, raw):
    """SHAP explanation of one applicant, with raw values attached for readable plots."""
    sv = explainer(scaled)
    values = np.asarray(sv.values)
    if values.ndim == 3:                                             # (rows, features, classes)
        values = values[:, :, 1]
    base = np.ravel(np.asarray(sv.base_values))[0]
    return shap.Explanation(values=values[0], base_values=base,
                            data=raw.iloc[0].values, feature_names=list(raw.columns))


# --- cached loaders --------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, scaler, columns, shap.TreeExplainer(model)


# --- user interface --------------------------------------------------------------------
EXAMPLES = {
    "Custom": None,
    "Lower-risk applicant": dict(loan_amnt=8000.0, int_rate=7.5, annual_inc=95000.0, dti=9.0, grade="A"),
    "Typical applicant": dict(loan_amnt=15000.0, int_rate=13.2, annual_inc=72000.0, dti=19.0, grade="C"),
    "Higher-risk applicant": dict(loan_amnt=28000.0, int_rate=27.0, annual_inc=38000.0, dti=30.0, grade="G"),
}


def apply_example():
    ex = EXAMPLES[st.session_state["example"]]
    if ex:
        for key, value in ex.items():
            st.session_state[key] = value


def main():
    st.set_page_config(page_title="Explainable Credit Risk", page_icon="💳", layout="wide")
    st.title("Explainable Credit Risk Prediction")
    st.caption("Minor project: Explainable Machine Learning for Credit Risk Prediction using SHAP "
               "(XGBoost model trained on LendingClub loans)")

    try:
        model, scaler, columns, explainer = load_artifacts()
    except Exception as exc:                                          # missing / incompatible files
        st.error("Could not load the model files from the `models/` folder. "
                 "Check that model.joblib, scaler.joblib and feature_columns.joblib are there "
                 "and that the library versions match requirements.txt.")
        st.exception(exc)
        st.stop()

    for key, value in EXAMPLES["Typical applicant"].items():          # first-run defaults
        st.session_state.setdefault(key, value)

    st.sidebar.header("Applicant details")
    st.sidebar.selectbox("Load an example", list(EXAMPLES), key="example", on_change=apply_example)
    loan_amnt = st.sidebar.number_input("Loan amount ($)", 1000.0, 40000.0, step=500.0, key="loan_amnt")
    int_rate = st.sidebar.number_input("Interest rate (%)", 5.0, 31.0, step=0.1, key="int_rate")
    annual_inc = st.sidebar.number_input("Annual income ($)", 4000.0, 1000000.0, step=1000.0, key="annual_inc")
    dti = st.sidebar.number_input("Debt-to-income ratio (%)", 0.0, 60.0, step=0.5, key="dti")
    grade = st.sidebar.selectbox("Loan grade", list(GRADE_MAP), key="grade")
    st.sidebar.caption("Interest rate and grade are normally closely linked "
                       "(correlation 0.97 in the training data).")

    with st.sidebar.expander("More details (optional)"):
        term = st.selectbox("Loan term (months)", TERMS, index=0)
        purpose = st.selectbox("Loan purpose", PURPOSES)
        home = st.selectbox("Home ownership", HOME)
        verification = st.selectbox("Income verification", VERIFICATION)
        emp_label = st.selectbox("Employment length", list(EMP_LENGTHS), index=len(EMP_LENGTHS) - 1)
        st.caption("Values left out (open accounts, revolving balance and utilisation, total accounts) "
                   "are set to the average of the training data.")

    raw = build_raw_row(columns, scaler, loan_amnt, int_rate, annual_inc, dti, grade,
                        term=term, purpose=purpose, home=home, verification=verification,
                        emp_length=EMP_LENGTHS[emp_label])
    scaled = scale_row(raw, scaler)
    prob = predict_default(model, scaled)
    expl = explain(explainer, scaled, raw)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Prediction")
        st.metric("Predicted default probability", f"{prob * 100:.1f}%")
        if prob >= THRESHOLD:
            st.error("Predicted: **likely to default** (probability at or above 50%)")
        else:
            st.success("Predicted: **likely to be repaid** (probability below 50%)")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption("The model was trained on a balanced sample (equal numbers of paid and defaulted "
                   "loans), so this probability is higher than real-world default rates. "
                   "Treat it as a risk score, not as an exact chance.")

        st.subheader("Main drivers")
        order = np.argsort(-np.abs(expl.values))[:5]
        for i in order:
            direction = "raises" if expl.values[i] > 0 else "lowers"
            st.write(f"- **{nice(expl.feature_names[i])}** {direction} the risk ({expl.values[i]:+.2f})")

    with right:
        st.subheader("SHAP explanation")
        try:
            plt.figure()
            shap.plots.waterfall(expl, max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
        except IndexError:                                            # shap / matplotlib version clash
            plt.close("all")
            top = np.argsort(-np.abs(expl.values))[:10][::-1]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh([nice(expl.feature_names[i]) for i in top], expl.values[top],
                    color=["#ff0051" if v > 0 else "#008bfb" for v in expl.values[top]])
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_xlabel("SHAP value (log-odds)")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        st.caption("Each bar shows how much one feature moved the prediction from the model's average "
                   "output (log-odds). Red pushes towards default, blue towards repayment. "
                   "The numbers on the left are the applicant's values (instalment is calculated "
                   "from the loan amount, rate and term).")

    with st.expander("About this app"):
        st.write("Four models (Logistic Regression, Decision Tree, Random Forest, XGBoost) were "
                 "compared on LendingClub data; XGBoost had the best AUC-ROC and is used here. "
                 "SHAP is used to explain each prediction. This is a student project for "
                 "demonstration only and must not be used for real lending decisions.")


if __name__ == "__main__":
    main()
 
