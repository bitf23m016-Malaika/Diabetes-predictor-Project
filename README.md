---
title: Diabetes Predictor API
emoji: 🩺
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
---

# 🩺 Diabetes Predictor — Backend API

**Logistic Regression model** trained on the Pima Indians Diabetes Dataset (UCI ML Repository, 768 patients).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/`        | Health check |
| POST | `/predict` | Get diabetes prediction |
| GET  | `/stats`   | Model performance metrics |

## `/predict` — Request Body (JSON)

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.47,
  "Age": 33
}
```

## `/predict` — Response

```json
{
  "prediction": 0,
  "probability": 28.4,
  "label": "Non-Diabetic",
  "risk_factors": []
}
```

## Model Performance
- **Accuracy**: ~82.5%
- **ROC-AUC**: ~0.90
- **F1 Score**: ~72%
- **CV Score (5-fold)**: ~84.5%

## Dataset
Pima Indians Diabetes Database — UCI ML Repository  
768 instances · 8 features · Binary classification
