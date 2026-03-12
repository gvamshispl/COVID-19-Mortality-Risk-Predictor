import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="COVID Mortality Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:60px;
    color:#1f77b4;
    font-weight:bolder;
}

.sub-title{
    text-align:center;
    font-size:30px;
    color:gray;
}

.result-box{
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.high-risk{
    background-color:#ffcccc;
    color:#990000;
}

.medium-risk{
    background-color:#fff3cd;
    color:#856404;
}

.low-risk{
    background-color:#d4edda;
    color:#155724;
}
.result-card {
    padding:25px;
    border-radius:12px;
    text-align:center;
    font-family:Arial;
    margin-top:20px;
}

.high-risk {
    background-color:#ffdddd;
    border-left:8px solid #ff0000;
}

.medium-risk {
    background-color:#fff4cc;
    border-left:8px solid #ffa500;
}

.low-risk {
    background-color:#ddffdd;
    border-left:8px solid #28a745;
}

.prediction-text{
    font-size:28px;
    font-weight:bold;
}

.prob-text{
    font-size:20px;
}

.info-box{
    background:blue;
    padding:15px;
    border-radius:8px;
    margin-top:15px;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.markdown('<p class="main-title">🩺 COVID-19 Mortality Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-based mortality risk prediction for COVID patients</p>', unsafe_allow_html=True)

st.write("")

# ---------------------------
# Input Layout
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    patient_type = st.selectbox("Patient Type", ["Outpatient", "Hospitalized"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.slider("Age", 0, 100, 40)
    icu = st.selectbox("ICU", ["Yes", "No"])
    intubed = st.selectbox("Intubed", ["Yes", "No"])

with col2:
    pneumonia = st.selectbox("Pneumonia", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    obesity = st.selectbox("Obesity", ["Yes", "No"])
    covid_res = st.selectbox("COVID Result", ["Positive", "Negative"])

# ---------------------------
# Convert Inputs
# ---------------------------
patient_type = 1 if patient_type=="Outpatient" else 2
sex = 1 if sex=="Female" else 2
icu = 1 if icu=="Yes" else 2
intubed = 1 if intubed=="Yes" else 2
pneumonia = 1 if pneumonia=="Yes" else 2
diabetes = 1 if diabetes=="Yes" else 2
hypertension = 1 if hypertension=="Yes" else 2
obesity = 1 if obesity=="Yes" else 2
covid_res = 1 if covid_res=="Positive" else 2

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔍 Predict Mortality Risk"):

    patient = [[
        patient_type,
        sex,
        age,
        icu,
        intubed,
        pneumonia,
        diabetes,
        hypertension,
        obesity,
        covid_res
    ]]

    patient = imputer.transform(patient)

    prob = model.predict_proba(patient)[0][1]
    prediction = model.predict(patient)[0]
    survival_prob = 1 - prob

    st.subheader("🧠 AI Prediction Result")

    # Risk styling
    if prob > 0.7:
        risk_class = "high-risk"
        risk_text = "🔴 HIGH RISK"
        message = "The model predicts a HIGH probability of death if the condition worsens."
    elif prob > 0.4:
        risk_class = "medium-risk"
        risk_text = "🟠 MODERATE RISK"
        message = "The model predicts a MODERATE risk. Patient should be monitored carefully."
    else:
        risk_class = "low-risk"
        risk_text = "🟢 LOW RISK"
        message = "The model predicts a LOW mortality risk."

    # Prediction label
    if prediction == 1:
        final_pred = "⚠️ Patient is at risk of dying"
    else:
        final_pred = "✅ Patient is likely to survive"

    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div class="prediction-text">{risk_text}</div>
        <p class="prob-text">Probability of Death: <b>{prob:.2f}</b></p>
        <p class="prob-text">Probability of Survival: <b>{survival_prob:.2f}</b></p>
        <h3>{final_pred}</h3>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    risk_score = round(prob * 100)
    st.metric(
        label="Mortality Risk Score",
        value=f"{risk_score}/100"
    )
    if prob > 0.75:
        st.error("🔴 Critical Risk — Immediate ICU monitoring recommended")
    elif prob > 0.5:
        st.warning("🟠 High Risk — Hospital monitoring required")
    elif prob > 0.25:
        st.info("🟡 Moderate Risk — Careful observation needed")
    else:
        st.success("🟢 Low Risk — Standard monitoring")

    st.subheader("Patient Risk Summary")

    st.write(f"Age: {age}")
    st.write(f"Pneumonia: {'Yes' if pneumonia==1 else 'No'}")
    st.write(f"Diabetes: {'Yes' if diabetes==1 else 'No'}")
    st.write(f"Hypertension: {'Yes' if hypertension==1 else 'No'}")

    st.progress(prob)
    import pandas as pd
    import matplotlib.pyplot as plt

    importance = model.feature_importances_
    features = [
    "patient_type",
    "sex",
    "age",
    "icu",
    "intubed",
    "pneumonia",
    "diabetes",
    "hypertension",
    "obesity",
    "covid_res"
    ]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Model Feature Importance")

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)
    st.markdown("""
    <div class="info-box">
    ℹ️ <b>About this AI Model</b><br>
    This machine learning model predicts the probability that a COVID-19 patient may die
    based on clinical factors such as age, pneumonia, ICU admission, and comorbidities.
    It helps identify high-risk patients early so healthcare providers can prioritize treatment.
    The prediction represents risk probability and not a guaranteed outcome.
    </div>
    """, unsafe_allow_html=True)