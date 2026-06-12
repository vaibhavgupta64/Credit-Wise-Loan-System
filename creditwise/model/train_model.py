"""
CreditWise - Model Training Script
===================================
Trains Logistic Regression, KNN, and Naive Bayes models on loan_approval_data.csv,
evaluates all three, and saves the best model + scaler to the model/ directory.

Usage:
    python model/train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "loan_approval_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLS_PATH  = os.path.join(MODEL_DIR, "feature_cols.pkl")

# ── 1. Load Data ──────────────────────────────────────────────
print("=" * 55)
print("  CreditWise — Model Training")
print("=" * 55)
print(f"\n📂 Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")

# ── 2. Handle Missing Values ──────────────────────────────────
print("\n🔧 Handling missing values...")

numerical_cols   = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target from imputation columns if present
if "Loan_Approved" in numerical_cols:
    numerical_cols.remove("Loan_Approved")
if "Loan_Approved" in categorical_cols:
    categorical_cols.remove("Loan_Approved")

num_imp = SimpleImputer(strategy="mean")
cat_imp = SimpleImputer(strategy="most_frequent")

df[numerical_cols]   = num_imp.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

print(f"   Missing values remaining: {df.isnull().sum().sum()}")

# ── 3. Drop Unnecessary Columns ───────────────────────────────
if "Applicant_ID" in df.columns:
    df = df.drop("Applicant_ID", axis=1)
    print("   Dropped: Applicant_ID")

# ── 4. Encode Target ──────────────────────────────────────────
print("\n🎯 Encoding target column...")
df["Loan_Approved"] = df["Loan_Approved"].map({"Yes": 1, "No": 0})

# Drop rows where target is still NaN after mapping
before = len(df)
df = df.dropna(subset=["Loan_Approved"])
dropped = before - len(df)
if dropped > 0:
    print(f"   ⚠️  Dropped {dropped} rows with missing Loan_Approved values")

df["Loan_Approved"] = df["Loan_Approved"].astype(int)
print(f"   Class distribution:\n{df['Loan_Approved'].value_counts().to_string()}")

# ── 5. Encode Categoricals ────────────────────────────────────
print("\n🔠 Encoding categorical features...")
df["Education_Level"] = df["Education_Level"].map(
    {"Graduate": 1, "Not Graduate": 0}
)

ohe_cols = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Gender", "Employer_Category"
]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
print(f"   Columns after encoding: {df.shape[1]}")

# ── 6. Feature Engineering ────────────────────────────────────
print("\n⚙️  Feature engineering...")
df["DTI_Ratio_sq"]          = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"]       = df["Credit_Score"] ** 2
df["Applicant_Income_log"]  = np.log1p(df["Applicant_Income"])
print("   Added: DTI_Ratio_sq, Credit_Score_sq, Applicant_Income_log")

# ── 7. Split Features & Target ────────────────────────────────
drop_cols = ["Loan_Approved", "Credit_Score", "DTI_Ratio"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df["Loan_Approved"]

feature_cols = X.columns.tolist()
print(f"\n📋 Total features used: {len(feature_cols)}")

# ── 8. Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✂️  Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 9. Scale Features ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 10. Train & Evaluate All Models ──────────────────────────
print("\n" + "=" * 55)
print("  Training & Evaluating Models")
print("=" * 55)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
}

results = {}

for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model":     clf,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }

    print(f"\n📌 {name}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   Confusion Matrix:\n{cm}")

# ── 11. Select Best Model (by Accuracy) ───────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_acc   = results[best_name]["accuracy"]

print("\n" + "=" * 55)
print(f"  ✅ Best Model: {best_name}")
print(f"     Accuracy : {best_acc:.4f}")
print("=" * 55)

# ── 12. Save Model, Scaler & Feature Columns ─────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

with open(COLS_PATH, "wb") as f:
    pickle.dump(feature_cols, f)

print(f"\n💾 Saved:")
print(f"   Model        → {MODEL_PATH}")
print(f"   Scaler       → {SCALER_PATH}")
print(f"   Feature cols → {COLS_PATH}")
print("\n🚀 Training complete! Run `streamlit run app.py` to launch.\n")