# 🏦 CreditWise — Loan Approval Prediction System

A machine learning web app that predicts **loan approval** based on income, credit score, DTI ratio & more. Trained on 3 models (Logistic Regression, Naive Bayes, KNN) with 88% accuracy. Built with Python, Scikit-learn & Streamlit.

---

## 📁 Project Structure

```
creditwise/
│
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── README.md
│
├── data/
│   └── loan_approval_data.csv  # Dataset (1000 records, 20 features)
│
├── model/
│   ├── train_model.py          # Training script
│   ├── best_model.pkl          # Saved trained model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_cols.pkl        # Feature column names
│
└── assets/                     # Images / logo
```

---

## ⚙️ Features

- ✅ Predicts loan approval with **confidence percentage**
- ✅ Trained on **3 ML models** — best one auto-selected
- ✅ Full **preprocessing pipeline** (imputation, encoding, scaling)
- ✅ **Feature engineering** (DTI², Credit Score², log income)
- ✅ Clean, responsive **Streamlit UI**
- ✅ Color-coded result — 🟢 Approved / 🔴 Rejected

---

## 🤖 Models Trained

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | **88%** | 78% | 84% | 81% |
| Naive Bayes | 86% | 81% | 70% | 75% |
| KNN (k=5) | 78% | 67% | 57% | 62% |

> ✅ **Best Model: Logistic Regression** (selected automatically)

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| Applicant_Income | Numeric | Monthly income of applicant |
| Coapplicant_Income | Numeric | Monthly income of co-applicant |
| Credit_Score | Numeric | Credit score (550–799) |
| DTI_Ratio | Numeric | Debt-to-income ratio |
| Loan_Amount | Numeric | Requested loan amount |
| Loan_Term | Numeric | Loan duration in months |
| Savings | Numeric | Applicant's savings |
| Collateral_Value | Numeric | Value of collateral |
| Age | Numeric | Applicant age |
| Dependents | Numeric | Number of dependents |
| Existing_Loans | Numeric | Number of existing loans |
| Employment_Status | Categorical | Salaried / Self-Employed |
| Marital_Status | Categorical | Married / Single / Divorced |
| Education_Level | Categorical | Graduate / Not Graduate |
| Gender | Categorical | Male / Female |
| Loan_Purpose | Categorical | Home / Car / Business / Personal / Education |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Employer_Category | Categorical | Government / MNC / Private / Self / Unemployed |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/creditwise.git
cd creditwise
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python model/train_model.py
```

This will generate `best_model.pkl`, `scaler.pkl`, and `feature_cols.pkl` inside the `model/` folder.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📦 Requirements

```
streamlit>=1.32.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **main file** as `app.py`
5. Click **Deploy** — get a public URL instantly!

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | ML models & preprocessing |
| Streamlit | Web app framework |
| Pickle | Model serialization |

---

## 📌 How It Works

```
User Input (18 fields)
        ↓
Feature Engineering (DTI², CreditScore², log income)
        ↓
One-Hot Encoding (categorical columns)
        ↓
StandardScaler (normalize features)
        ↓
Logistic Regression Model
        ↓
Prediction → Approved ✅ / Rejected ❌ + Confidence %
```

---

## 👨‍💻 Author

**Vaibhav**
- GitHub: [@vaibhavgupta64](https://github.com/vaibhavgupta64)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
