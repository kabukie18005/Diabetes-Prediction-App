# 🩺 Diabetes Risk Prediction App

A multi-page Streamlit web application that predicts diabetes risk using an ensemble
of three machine learning models: **K-Nearest Neighbours**, **Logistic Regression**,
and an **Artificial Neural Network**.

Built as a final year project for BSc Administration (Business Analytics).

---

## Features

| Page | Description |
|---|---|
| **Home** | Enter health details via sliders or manual input and get an instant risk score with a visual speedometer |
| **Data Exploration** | Visualise the dataset — distributions, correlations, and class balance |
| **Batch Prediction** | Upload a CSV to get predictions for multiple patients; download results |
| **Model Performance** | Compare accuracy, precision, recall, F1, confusion matrices, and ROC curves for all three models |
| **About** | Dataset info, model descriptions, and project team |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-app.git
cd diabetes-prediction-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

> Make sure `diabetes.csv` is in the **same folder** as `app.py`.

---

## Dataset

**Pima Indian Diabetes Database** — UCI Machine Learning Repository

768 female patients, 8 features, binary outcome (diabetic / not diabetic).

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic risk score |
| Age | Age in years |

---

## Models

- **KNN** — classifies based on the k nearest training examples
- **Logistic Regression** — linear probabilistic classifier
- **ANN** — two hidden layers (64 → 32 neurons) with dropout regularisation

The final risk score is the **average probability** from all three models.

---

## Disclaimer

This app is for **educational and portfolio purposes only** and is not intended
for clinical or medical use. Always consult a qualified healthcare professional.
