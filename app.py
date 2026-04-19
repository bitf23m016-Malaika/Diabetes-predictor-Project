"""
Diabetes Predictor — Flask Backend
Deployed on : Hugging Face Spaces (Docker SDK)
Frontend    : GitHub Pages (CORS enabled)
"""

import os, pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow GitHub Pages frontend to call this API

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load Artifacts ─────────────────────────────────────────────────────────────
def load():
    with open(os.path.join(BASE,"model","logistic_model.pkl"),"rb") as f: m = pickle.load(f)
    with open(os.path.join(BASE,"model","scaler.pkl"),         "rb") as f: s = pickle.load(f)
    with open(os.path.join(BASE,"model","stats.pkl"),          "rb") as f: t = pickle.load(f)
    return m, s, t

try:
    model, scaler, stats = load()
    print("[OK] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] {e} — run train.py first.")
    model = scaler = stats = None

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]

LIMITS = {
    "Pregnancies"             : (0,   20),
    "Glucose"                 : (40,  300),
    "BloodPressure"           : (20,  140),
    "SkinThickness"           : (0,   110),
    "Insulin"                 : (0,   1000),
    "BMI"                     : (10,  80),
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age"                     : (1,   120),
}

# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Diabetes Predictor API is running."})

# ── Predict ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 500

    data = request.get_json(force=True)

    # Extract & validate
    try:
        feats = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    for i, col in enumerate(FEATURES):
        lo, hi = LIMITS[col]
        if not (lo <= feats[i] <= hi):
            return jsonify({"error": f"{col} = {feats[i]} is out of range [{lo}, {hi}]."}), 400

    X  = np.array(feats).reshape(1, -1)
    Xs = scaler.transform(X)

    pred = int(model.predict(Xs)[0])
    prob = float(model.predict_proba(Xs)[0][1])

    # Risk factor analysis
    risks = []
    if feats[1] > 125:
        risks.append({"label":"High Glucose",         "value":f"{int(feats[1])} mg/dL","level":"high"})
    elif feats[1] > 100:
        risks.append({"label":"Borderline Glucose",   "value":f"{int(feats[1])} mg/dL","level":"medium"})
    if feats[5] > 30:
        risks.append({"label":"High BMI",             "value":str(feats[5]),            "level":"high"})
    elif feats[5] > 25:
        risks.append({"label":"Overweight BMI",       "value":str(feats[5]),            "level":"medium"})
    if feats[7] > 45:
        risks.append({"label":"Advanced Age",         "value":f"{int(feats[7])} yrs",   "level":"high"})
    elif feats[7] > 35:
        risks.append({"label":"Age Factor",           "value":f"{int(feats[7])} yrs",   "level":"medium"})
    if feats[6] > 0.8:
        risks.append({"label":"High Pedigree Score",  "value":str(feats[6]),            "level":"high"})
    if feats[0] > 5:
        risks.append({"label":"Multiple Pregnancies", "value":str(int(feats[0])),       "level":"medium"})

    return jsonify({
        "prediction"  : pred,
        "probability" : round(prob * 100, 1),
        "label"       : "Diabetic" if pred == 1 else "Non-Diabetic",
        "risk_factors": risks,
    })

# ── Stats ──────────────────────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def get_stats():
    if stats is None:
        return jsonify({"error": "Stats not available."}), 500
    return jsonify(stats)

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))   # HuggingFace uses 7860
    app.run(host="0.0.0.0", port=port, debug=False)
