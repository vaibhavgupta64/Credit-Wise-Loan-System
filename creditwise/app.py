import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise – Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-size: 1.1rem; font-weight: 600;
        width: 100%; margin-top: 1rem;
    }
    .stButton>button:hover { opacity: 0.9; }
    .result-approved {
        background: #dcfce7; border-left: 5px solid #16a34a;
        border-radius: 8px; padding: 1.2rem; margin-top: 1rem;
    }
    .result-rejected {
        background: #fee2e2; border-left: 5px solid #dc2626;
        border-radius: 8px; padding: 1.2rem; margin-top: 1rem;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Train model (cached) ──────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "Applicant_Income":    np.random.uniform(2000, 20000, n),
        "Coapplicant_Income":  np.random.uniform(0, 10000, n),
        "Age":                 np.random.randint(21, 60, n).astype(float),
        "Dependents":          np.random.randint(0, 4, n).astype(float),
        "Existing_Loans":      np.random.randint(0, 5, n).astype(float),
        "Savings":             np.random.uniform(0, 20000, n),
        "Collateral_Value":    np.random.uniform(0, 50000, n),
        "Loan_Amount":         np.random.uniform(1000, 40000, n),
        "Loan_Term":           np.random.choice([12, 24, 36, 48, 60, 72, 84], n).astype(float),
        "Education_Level":     np.random.randint(0, 2, n),
        "Employment_Status":   np.random.choice(["Salaried", "Self-Employed"], n),
        "Marital_Status":      np.random.choice(["Married", "Single", "Divorced"], n),
        "Loan_Purpose":        np.random.choice(["Home", "Car", "Business", "Personal", "Education"], n),
        "Property_Area":       np.random.choice(["Urban", "Semiurban", "Rural"], n),
        "Gender":              np.random.choice(["Male", "Female"], n),
        "Employer_Category":   np.random.choice(["Government", "MNC", "Private", "Unemployed", "Self"], n),
        "DTI_Ratio":           np.random.uniform(0.1, 0.6, n),
        "Credit_Score":        np.random.randint(550, 800, n).astype(float),
    })

    # Feature engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

    # Simulate target
    score = (
        (df["Credit_Score"] - 550) / 250 * 0.4
        + (1 - df["DTI_Ratio"]) * 0.3
        + (df["Savings"] / 20000) * 0.15
        + (df["Applicant_Income"] / 20000) * 0.15
    )
    df["Loan_Approved"] = (score + np.random.normal(0, 0.1, n) > 0.55).astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=[
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Gender", "Employer_Category"
    ], drop_first=False)

    feature_cols = [c for c in df.columns if c != "Loan_Approved"]

    X = df[feature_cols]
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_sc, y_train)

    return model, scaler, feature_cols

model, scaler, feature_cols = train_model()

def predict(inputs: dict):
    row = pd.DataFrame([inputs])

    # Feature engineering
    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    row["Applicant_Income_log"] = np.log1p(row["Applicant_Income"])

    # One-hot encode categoricals
    row = pd.get_dummies(row, columns=[
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Gender", "Employer_Category"
    ])

    # Align columns with training features
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]

    scaled = scaler.transform(row)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]
    return pred, prob

# ── UI ────────────────────────────────────────────────────────
st.title("🏦 CreditWise")
st.markdown("#### Loan Approval Prediction System")
st.markdown("Fill in the applicant details below to get an instant decision.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**👤 Personal Information**")
    age = st.slider("Age", 21, 59, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

with col2:
    st.markdown("**💼 Employment & Income**")
    employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
    employer_cat = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Self", "Unemployed"])
    applicant_income = st.number_input("Applicant Monthly Income (₹)", 2000, 20000, 8000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", 0, 10000, 2000, step=500)

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.markdown("**🏠 Loan Details**")
    loan_amount = st.number_input("Loan Amount (₹)", 1000, 40000, 15000, step=1000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Business", "Personal", "Education"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col4:
    st.markdown("**📊 Financial Profile**")
    credit_score = st.slider("Credit Score", 550, 799, 680)
    dti_ratio = st.slider("DTI Ratio (Debt-to-Income)", 0.10, 0.60, 0.35, step=0.01)
    savings = st.number_input("Savings (₹)", 0, 20000, 5000, step=500)
    collateral = st.number_input("Collateral Value (₹)", 0, 50000, 15000, step=1000)
    existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3, 4])

st.divider()

if st.button("🔍 Predict Loan Approval"):
    inputs = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": float(age),
        "Dependents": float(dependents),
        "Existing_Loans": float(existing_loans),
        "Savings": float(savings),
        "Collateral_Value": float(collateral),
        "Loan_Amount": float(loan_amount),
        "Loan_Term": float(loan_term),
        "Education_Level": 1 if education == "Graduate" else 0,
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_cat,
        "DTI_Ratio": dti_ratio,
        "Credit_Score": float(credit_score),
    }

    pred, prob = predict(inputs)
    approve_pct = round(prob[1] * 100, 1)
    reject_pct = round(prob[0] * 100, 1)

    if pred == 1:
        st.markdown(f"""
        <div class="result-approved">
            <h3 style="color:#16a34a; margin:0">✅ Loan APPROVED</h3>
            <p style="margin:0.5rem 0 0">Approval confidence: <strong>{approve_pct}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <h3 style="color:#dc2626; margin:0">❌ Loan REJECTED</h3>
            <p style="margin:0.5rem 0 0">Rejection confidence: <strong>{reject_pct}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><b>Credit Score</b><br><span style="font-size:1.4rem;color:#1e3a8a">{credit_score}</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><b>DTI Ratio</b><br><span style="font-size:1.4rem;color:#1e3a8a">{dti_ratio:.2f}</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><b>Approval Probability</b><br><span style="font-size:1.4rem;color:#1e3a8a">{approve_pct}%</span></div>', unsafe_allow_html=True)

st.divider()
st.caption("CreditWise · Powered by Logistic Regression · Built with Streamlit")
