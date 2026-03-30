import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MediScope - Heart Disease Risk",
    page_icon="",
    layout="wide"
)

# Load model and scaler (cached - runs once per session)
@st.cache_resource
def load_models():
    rf_model = joblib.load('heart_model.pkl')
    lr_model = joblib.load('heart_lr.pkl')
    scaler = joblib.load('heart_scaler.pkl')
    return rf_model, lr_model, scaler

rf_model, lr_model, scaler = load_models()

# Application header
st.title("MediScope: Heart Disease Risk Assessment")
st.markdown(
    "Enter the patient's clinical measurements in the sidebar. "
    "The model will estimate the probability of coronary heart disease "
    "based on 13 clinical variables from the UCI Heart Disease dataset. "
    "This tool is for research and educational purposes only."
)
st.divider()

st.info("Complete the patient form in the sidebar and press **Assess Risk** to generate a prediction.")

# Sidebar: Patient Input Form
st.sidebar.header("Patient Clinical Data")

age      = st.sidebar.slider("Age (years)", min_value=20, max_value=80, value=55)
sex      = st.sidebar.selectbox("Sex", options=["Male", "Female"])
sex_val  = 1 if sex == "Male" else 0

cp_labels = {
    "Typical angina (1)": 1,
    "Atypical angina (2)": 2,
    "Non-anginal pain (3)": 3,
    "Asymptomatic (4)": 4,
}
cp = st.sidebar.selectbox("Chest Pain Type", list(cp_labels.keys()))
cp_val = cp_labels[cp]

trestbps = st.sidebar.number_input("Resting Blood Pressure (mmHg)", min_value=80,  max_value=220, value=130)
chol     = st.sidebar.number_input("Serum Cholesterol (mg/dl)",     min_value=100, max_value=600, value=245)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
fbs_val  = 1 if fbs == "Yes (1)" else 0

restecg_labels = {
    "Normal (0)": 0,
    "ST-T wave abnormality (1)": 1,
    "Left ventricular hypertrophy (2)": 2,
}
restecg     = st.sidebar.selectbox("Resting ECG Results", list(restecg_labels.keys()))
restecg_val = restecg_labels[restecg]

thalach  = st.sidebar.slider("Max Heart Rate Achieved (bpm)",  min_value=60,  max_value=220, value=150)
exang    = st.sidebar.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"])
exang_val = 1 if exang == "Yes (1)" else 0

oldpeak  = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

slope_labels = {"Upsloping (1)": 1, "Flat (2)": 2, "Downsloping (3)": 3}
slope    = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", list(slope_labels.keys()))
slope_val = slope_labels[slope]

ca       = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

thal_labels = {"Normal (3)": 3, "Fixed defect (6)": 6, "Reversible defect (7)": 7}
thal     = st.sidebar.selectbox("Thalassaemia", list(thal_labels.keys()))
thal_val = thal_labels[thal]

# Assemble the input row in the same column order as training
input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                         restecg_val, thalach, exang_val, oldpeak,
                         slope_val, ca, thal_val]])

predict_btn = st.sidebar.button("Assess Risk", type="primary", use_container_width=True)

# Prediction and Output Display
feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

if predict_btn:
    input_scaled = scaler.transform(input_data)

    # Probabilidades dos dois modelos
    prob_rf = rf_model.predict_proba(input_scaled)[0, 1]
    prob_lr = lr_model.predict_proba(input_scaled)[0, 1]

    risk_pct_rf = prob_rf * 100
    risk_pct_lr = prob_lr * 100

    # Usar o RF como referência para o tier
    risk_pct = risk_pct_rf

    # Risk tier labels and colours
    if risk_pct < 30:
        tier, colour = "Low Risk", "green"
    elif risk_pct < 60:
        tier, colour = "Moderate Risk", "orange"
    else:
        tier, colour = "High Risk", "red"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Random Forest")
        st.metric(
            label="Predicted Probability of Heart Disease",
            value=f"{risk_pct_rf:.1f}%"
        )
        st.markdown(f"**Classification (RF): :{colour}[{tier}]**")

    with col2:
        st.subheader("Logistic Regression")
        st.metric(
            label="Predicted Probability of Heart Disease",
            value=f"{risk_pct_lr:.1f}%"
        )

    st.caption(
        "Thresholds: Low < 30% | Moderate 30–60% | High > 60%. "
        "These are illustrative boundaries for educational use, "
        "not validated clinical cut-offs."
    )

    # Linha extra: qual o modelo mais confiante e por quanto
    if prob_rf > prob_lr:
        diff = (prob_rf - prob_lr) * 100
        st.write(f"O Random Forest é mais confiante por {diff:.1f} pontos percentuais.")
    elif prob_lr > prob_rf:
        diff = (prob_lr - prob_rf) * 100
        st.write(f"A Regressão Logística é mais confiante por {diff:.1f} pontos percentuais.")
    else:
        st.write("Ambos os modelos têm exatamente a mesma probabilidade prevista.")

    # Top contributing features continua, mas agora com rf_model
    st.subheader("Top Contributing Features")
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(7)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 7 Features (Random Forest)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Patient summary table
    st.subheader("Patient Data Summary")
    summary = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_data[0]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
