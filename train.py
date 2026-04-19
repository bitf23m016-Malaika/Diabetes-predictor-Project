"""
Diabetes Predictor — Training Script
Dataset : Pima Indians Diabetes (UCI ML Repository / Kaggle)
Model   : Logistic Regression (scikit-learn)
Run     : python train.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "dataset", "diabetes.csv")
MDL_DIR   = os.path.join(BASE, "model")
os.makedirs(MDL_DIR, exist_ok=True)

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
TARGET   = "Outcome"

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded {df.shape[0]} rows × {df.shape[1]} cols")
print(df[TARGET].value_counts().to_string(), "\n")

# ── 2. Preprocess ──────────────────────────────────────────────────────────────
for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
    median = df[col][df[col] != 0].median()
    df[col] = df[col].replace(0, median)

X = df[FEATURES].values
y = df[TARGET].values

# ── 3. Split + Scale ───────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 4. Train ───────────────────────────────────────────────────────────────────
model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
model.fit(X_tr_s, y_tr)

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_te_s)
y_prob = model.predict_proba(X_te_s)[:, 1]

acc   = float(accuracy_score(y_te, y_pred))
prec  = float(precision_score(y_te, y_pred))
rec   = float(recall_score(y_te, y_pred))
f1    = float(f1_score(y_te, y_pred))
auc   = float(roc_auc_score(y_te, y_prob))
cm    = confusion_matrix(y_te, y_pred).tolist()

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvs   = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring="accuracy")

print("="*50)
print(f"  Accuracy   : {acc*100:.2f}%")
print(f"  Precision  : {prec*100:.2f}%")
print(f"  Recall     : {rec*100:.2f}%")
print(f"  F1 Score   : {f1*100:.2f}%")
print(f"  ROC-AUC    : {auc:.4f}")
print(f"  CV (5-fold): {cvs.mean()*100:.2f}% ± {cvs.std()*100:.2f}%")
print(f"  Conf Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
print("="*50)
print(classification_report(y_te, y_pred, target_names=["No Diabetes","Diabetes"]))

# ── 6. Save (all native Python types — no numpy floats) ───────────────────────
stats = {
    "accuracy"  : round(acc*100, 2),
    "precision" : round(prec*100, 2),
    "recall"    : round(rec*100, 2),
    "f1"        : round(f1*100, 2),
    "roc_auc"   : round(auc, 4),
    "cv_mean"   : round(float(cvs.mean())*100, 2),
    "cv_std"    : round(float(cvs.std())*100, 2),
    "confusion_matrix": cm,
}

with open(os.path.join(MDL_DIR,"logistic_model.pkl"),"wb") as f: pickle.dump(model,  f)
with open(os.path.join(MDL_DIR,"scaler.pkl"),         "wb") as f: pickle.dump(scaler, f)
with open(os.path.join(MDL_DIR,"stats.pkl"),          "wb") as f: pickle.dump(stats,  f)

print("[SAVED] model/ → logistic_model.pkl  scaler.pkl  stats.pkl")
print("[DONE]  Run:  python app.py")
