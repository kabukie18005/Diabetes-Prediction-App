import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Minimal global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
h1, h2, h3 { color: #1c4e80; }
.stButton > button {
    background-color: #1c4e80;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover { background-color: #163d66; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    return df


@st.cache_data
def preprocess(df):
    imputer = SimpleImputer(strategy="mean")
    df_imp = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    X = df_imp.drop("Outcome", axis=1)
    y = df_imp["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_test),
        y_train, y_test,
        scaler, X.columns.tolist()
    )


# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource
def train_knn(_X_train, _y_train, k=5):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(_X_train, _y_train)
    return m


@st.cache_resource
def train_lr(_X_train, _y_train):
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(_X_train, _y_train)
    return m


@st.cache_resource
def train_ann(_X_train, _y_train, epochs=50, dropout=0.2):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(_X_train.shape[1],)),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(_X_train, _y_train, epochs=epochs, batch_size=16, verbose=0)
    return model


# ── Speedometer HTML ──────────────────────────────────────────────────────────
def speedometer(score: float):
    if score < 0.35:
        color, label = "#28a745", "Low Risk"
    elif score < 0.65:
        color, label = "#ffc107", "Medium Risk"
    else:
        color, label = "#dc3545", "High Risk"

    angle = score * 180 - 90
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0;">
      <div style="position:relative; width:220px; height:110px; margin:auto;
                  border-radius:110px 110px 0 0; overflow:hidden; background:#eee;">
        <div style="position:absolute; inset:0;
          background:conic-gradient(#28a745 0deg 60deg,#ffc107 60deg 120deg,#dc3545 120deg 180deg);"></div>
        <div style="position:absolute; top:100%; left:50%;
          transform:translate(-50%,-100%);
          width:200px; height:100px;
          background:white; border-radius:100px 100px 0 0;"></div>
        <div style="position:absolute; top:100%; left:50%;
          transform-origin:bottom center;
          transform:translate(-50%,-100%) rotate({angle}deg);
          width:3px; height:100px;
          background:#333; border-radius:3px 3px 0 0;"></div>
      </div>
      <p style="font-size:1.4rem; font-weight:700; color:{color}; margin-top:10px;">{label}</p>
      <p style="font-size:1rem; color:#555;">Risk score: <b>{score:.1%}</b></p>
    </div>
    """, unsafe_allow_html=True)


# ── Health recommendations ────────────────────────────────────────────────────
def recommendations(score: float):
    st.markdown("---")
    st.subheader("Health Recommendations")
    st.caption("⚠️ This tool is not a substitute for professional medical advice.")
    if score < 0.35:
        st.success("**Low risk.** Keep up your healthy habits!")
        st.markdown(
            "- Maintain a balanced diet rich in vegetables and whole grains.\n"
            "- Stay physically active — at least 30 minutes most days.\n"
            "- Schedule annual check-ups with your doctor."
        )
    elif score < 0.65:
        st.warning("**Medium risk.** Some risk factors detected — take action now.")
        st.markdown(
            "- Reduce sugar and processed food intake.\n"
            "- Aim for moderate weight loss if overweight (even 5–7% helps).\n"
            "- Watch for symptoms: frequent urination, unusual thirst, fatigue.\n"
            "- Consider a formal blood glucose test."
        )
    else:
        st.error("**High risk.** Please consult a healthcare professional soon.")
        st.markdown(
            "- Book an appointment for an HbA1c blood test immediately.\n"
            "- Work with a dietitian to create a personalised meal plan.\n"
            "- Your doctor can recommend a safe exercise programme.\n"
            "- Monitor your blood sugar regularly."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home (Single Prediction)
# ══════════════════════════════════════════════════════════════════════════════
def page_home(X_train_sc, y_train, scaler, feature_cols):
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Enter your health details below to get an instant risk assessment powered by three machine learning models.")

    method = st.radio("Input method", ["Sliders", "Manual entry"], horizontal=True)

    col1, col2 = st.columns(2)
    fields = {}

    inputs = [
        ("Pregnancies",             0,   20,   1,    0,    "Number of pregnancies"),
        ("Glucose",                 0,  200, 100,    0,    "Plasma glucose level (mg/dL)"),
        ("BloodPressure",           0,  140,  70,    0,    "Diastolic blood pressure (mm Hg)"),
        ("SkinThickness",           0,  100,  20,    0,    "Skin fold thickness (mm)"),
        ("Insulin",                 0,  900,  80,    0,    "2-hour serum insulin (μU/mL)"),
        ("BMI",                   0.0, 70.0, 25.0,  0.1,  "Body mass index"),
        ("DiabetesPedigreeFunction",0.0, 2.5,  0.5, 0.01, "Diabetes pedigree function"),
        ("Age",                     1,  120,  30,    1,    "Age (years)"),
    ]

    for i, (name, mn, mx, default, step, label) in enumerate(inputs):
        col = col1 if i < 4 else col2
        with col:
            if method == "Sliders":
                if isinstance(step, float):
                    fields[name] = st.slider(label, float(mn), float(mx), float(default), step)
                else:
                    fields[name] = st.slider(label, int(mn), int(mx), int(default), int(step) if step else 1)
            else:
                if isinstance(step, float):
                    fields[name] = st.number_input(label, min_value=float(mn), max_value=float(mx),
                                                    value=float(default), step=step)
                else:
                    fields[name] = st.number_input(label, min_value=int(mn), max_value=int(mx),
                                                    value=int(default), step=max(1, int(step)))

    if st.button("Analyse My Risk"):
        user_arr = np.array([[fields[f] for f in feature_cols]])
        user_sc  = scaler.transform(user_arr)

        knn = train_knn(X_train_sc, y_train)
        lr  = train_lr(X_train_sc, y_train)
        ann = train_ann(X_train_sc, y_train)

        p_knn = knn.predict_proba(user_sc)[0, 1]
        p_lr  = lr.predict_proba(user_sc)[0, 1]
        p_ann = float(ann.predict(user_sc, verbose=0)[0][0])
        score = (p_knn + p_lr + p_ann) / 3

        st.markdown("---")
        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KNN",                 f"{p_knn:.1%}")
        c2.metric("Logistic Regression", f"{p_lr:.1%}")
        c3.metric("Neural Network",      f"{p_ann:.1%}")
        c4.metric("Ensemble Score",      f"{score:.1%}")

        speedometer(score)
        recommendations(score)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Exploratory Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
def page_eda(df):
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Explore the Pima Indian Diabetes dataset used to train the models.")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Class balance")
    counts = df["Outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Not Diabetic (0)", "Diabetic (1)"], counts.values,
           color=["#1c4e80", "#e05c1c"])
    ax.set_ylabel("Count")
    ax.set_title("Outcome distribution")
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature distributions")
    num_cols = [c for c in df.columns if c != "Outcome"]
    cols = st.columns(2)
    for i, col_name in enumerate(num_cols):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.histplot(df[col_name].dropna(), kde=True, ax=ax, color="#1c4e80")
        ax.set_title(col_name)
        cols[i % 2].pyplot(fig)
        plt.close()

    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Feature correlations")
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature vs Outcome (box plots)")
    cols2 = st.columns(2)
    for i, col_name in enumerate(num_cols):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        df.boxplot(column=col_name, by="Outcome", ax=ax,
                   boxprops=dict(color="#1c4e80"))
        ax.set_title(col_name)
        ax.set_xlabel("Outcome (0 = No, 1 = Yes)")
        plt.suptitle("")
        cols2[i % 2].pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
def page_batch(X_train_sc, y_train, scaler, feature_cols):
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV file with the same columns as the training data to get predictions for multiple patients at once.")
    st.info("Required columns: " + ", ".join(feature_cols))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        return

    batch_df = pd.read_csv(uploaded)
    st.subheader("Uploaded data")
    st.dataframe(batch_df.head(), use_container_width=True)

    # Check columns
    missing = [c for c in feature_cols if c not in batch_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Please check your file.")
        return

    # Preprocess
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_with_zeros:
        if c in batch_df.columns:
            batch_df[c] = batch_df[c].replace(0, np.nan)

    imputer = SimpleImputer(strategy="mean")
    batch_imp = pd.DataFrame(imputer.fit_transform(batch_df[feature_cols]),
                             columns=feature_cols)
    batch_sc = scaler.transform(batch_imp)

    # Train models
    knn = train_knn(X_train_sc, y_train)
    lr  = train_lr(X_train_sc, y_train)
    ann = train_ann(X_train_sc, y_train)

    p_knn = knn.predict_proba(batch_sc)[:, 1]
    p_lr  = lr.predict_proba(batch_sc)[:, 1]
    p_ann = ann.predict(batch_sc, verbose=0).flatten()
    ensemble = (p_knn + p_lr + p_ann) / 3

    results = batch_df.copy()
    results["KNN_Probability"]  = p_knn.round(3)
    results["LR_Probability"]   = p_lr.round(3)
    results["ANN_Probability"]  = p_ann.round(3)
    results["Ensemble_Score"]   = ensemble.round(3)
    results["Prediction"] = np.where(ensemble >= 0.5, "Diabetic", "Not Diabetic")

    st.subheader("Prediction results")
    st.dataframe(results, use_container_width=True)

    # Summary
    diabetic_count = (results["Prediction"] == "Diabetic").sum()
    st.markdown(f"**{diabetic_count}** of **{len(results)}** patients flagged as Diabetic "
                f"({diabetic_count/len(results):.1%})")

    # Download button
    csv_bytes = results.to_csv(index=False).encode()
    st.download_button(
        label="Download results as CSV",
        data=csv_bytes,
        file_name="diabetes_predictions.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
def page_performance(X_train_sc, X_test_sc, y_train, y_test):
    st.title("📈 Model Performance")
    st.markdown("Compare how each model performs on the held-out test set. Adjust hyperparameters in the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameters")
    k        = st.sidebar.slider("KNN — neighbours (k)", 1, 15, 5)
    epochs   = st.sidebar.slider("ANN — epochs", 10, 100, 50, 10)
    dropout  = st.sidebar.slider("ANN — dropout rate", 0.1, 0.5, 0.2, 0.05)

    with st.spinner("Training models…"):
        knn = train_knn(X_train_sc, y_train, k=k)
        lr  = train_lr(X_train_sc, y_train)
        ann = train_ann(X_train_sc, y_train, epochs=epochs, dropout=dropout)

    # Predictions
    y_knn  = knn.predict(X_test_sc)
    y_lr   = lr.predict(X_test_sc)
    y_ann  = (ann.predict(X_test_sc, verbose=0).flatten() > 0.5).astype(int)
    p_knn  = knn.predict_proba(X_test_sc)[:, 1]
    p_lr   = lr.predict_proba(X_test_sc)[:, 1]
    p_ann  = ann.predict(X_test_sc, verbose=0).flatten()

    def metrics(y_true, y_pred):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred),  3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall":    round(recall_score(y_true, y_pred),     3),
            "F1 Score":  round(f1_score(y_true, y_pred),         3),
        }

    # Metrics table
    st.subheader("Metrics comparison")
    df_metrics = pd.DataFrame({
        "KNN":                  metrics(y_test, y_knn),
        "Logistic Regression":  metrics(y_test, y_lr),
        "Neural Network":       metrics(y_test, y_ann),
    }).T
    st.dataframe(df_metrics.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    st.markdown("---")

    # Confusion matrices
    st.subheader("Confusion matrices")
    c1, c2, c3 = st.columns(3)
    for col, name, y_pred in zip([c1, c2, c3],
                                  ["KNN", "Logistic Regression", "Neural Network"],
                                  [y_knn, y_lr, y_ann]):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                    cmap="Blues", ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        col.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ROC curves
    st.subheader("ROC curves")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, probs in [("KNN", p_knn), ("Logistic Regression", p_lr), ("ANN", p_ann)]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — all models")
    ax.legend()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — About
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.title("ℹ️ About This App")
    st.markdown("""
    ### Diabetes Risk Prediction Hub

    This app uses machine learning to predict diabetes risk based on the
    **Pima Indian Diabetes Dataset** (UCI Machine Learning Repository).

    #### Models used
    | Model | Description |
    |---|---|
    | K-Nearest Neighbours (KNN) | Classifies based on similarity to neighbours |
    | Logistic Regression | Linear probabilistic classifier |
    | Artificial Neural Network (ANN) | Deep learning with two hidden layers |

    The **ensemble score** averages the probability outputs of all three models
    for a more robust prediction.

    #### Dataset features
    | Feature | Description |
    |---|---|
    | Pregnancies | Number of pregnancies |
    | Glucose | Plasma glucose concentration (mg/dL) |
    | Blood Pressure | Diastolic blood pressure (mm Hg) |
    | Skin Thickness | Triceps skin fold thickness (mm) |
    | Insulin | 2-hour serum insulin (μU/mL) |
    | BMI | Body mass index |
    | Diabetes Pedigree Function | Genetic risk score |
    | Age | Age in years |

    #### Disclaimer
    > This application is for **educational and portfolio purposes only**.
    > It is not a medical device and should not be used for clinical diagnosis.
    > Always consult a qualified healthcare professional.

    ---
    *BSc Administration (Business Analytics) — Final Year Project*
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.sidebar.title("🩺 Diabetes App")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Data Exploration", "Batch Prediction", "Model Performance", "About"]
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Pima Indian Diabetes Dataset | UCI ML Repository")

    # Load and prepare data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("diabetes.csv not found. Please place it in the same folder as app.py.")
        st.stop()

    X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols = preprocess(df)

    if page == "Home":
        page_home(X_train_sc, y_train, scaler, feature_cols)
    elif page == "Data Exploration":
        page_eda(df)
    elif page == "Batch Prediction":
        page_batch(X_train_sc, y_train, scaler, feature_cols)
    elif page == "Model Performance":
        page_performance(X_train_sc, X_test_sc, y_train, y_test)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
