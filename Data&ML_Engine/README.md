# EduSight – Student Dropout Risk Prediction System

EduSight is an AI-powered student dropout risk prediction and intervention recommendation system designed to help schools identify at-risk students early and recommend targeted support strategies.

The system consists of three integrated engines:

- **Risk Engine** – Calculates student dropout risk scores using weighted factors.
- **Machine Learning Engine** – Trains predictive models using historical student performance data.
- **Recommendation Engine** – Generates personalized intervention recommendations.
- **Integration Test Suite** – Validates the complete pipeline.

---

# Project Structure

```
EduSight/
│
├── risk_engine.py
├── ml_engine.py
├── recommendation_engine.py
├── test_edusight.py
│
├── Maths.csv
├── Portuguese.csv
│
├── requirements.txt
└── README.md
```

---

# Features

## Risk Prediction Engine

Evaluates student dropout risk based on:

- Attendance rate
- Academic performance
- Socioeconomic status
- Family support
- Historical trend analysis

Outputs:

- Risk score (0–100)
- Risk level (Low / Medium / High)
- 3-month dropout probability
- 6-month dropout probability
- Explanation of risk factors

---

## Machine Learning Engine

Uses the UCI Student Performance Dataset to train predictive models.

Models:

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression Blending Layer

Capabilities:

- Dropout prediction
- At-risk prediction
- Grade prediction
- Batch inference
- Feature importance analysis
- Model persistence using Joblib

---

## Recommendation Engine

Generates intervention plans such as:

- Attendance improvement programs
- Academic tutoring
- Counselling support
- Welfare assistance
- Family engagement plans
- Ongoing monitoring

Recommendations are prioritized by:

- Risk severity
- Expected impact
- Simulation outcomes

---

# Dataset

This project uses the UCI Student Performance Dataset.

The ML engine automatically:

- Loads CSV or Excel files
- Performs feature engineering
- Creates dropout labels
- Creates at-risk labels
- Encodes categorical features

---

# Running Tests

## Test A — Rule-Based Engine Only

No dataset required.

```bash
python test_edusight.py rule
```

---

## Test B — Train Machine Learning Model

Using Mathematics dataset:

```bash
python test_edusight.py ml data/Maths.csv
```

Using both datasets:

```bash
python test_edusight.py ml data/Maths.csv data/Portuguese.csv
```
---

## Test C — Full System Test

Runs complete integration testing.

```bash
python test_edusight.py all
```

This validates:

- Risk Engine
- Recommendation Engine
- Simulation Engine
- ML Integration
- End-to-End Pipeline
