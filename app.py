import streamlit as st
import pandas as pd
import joblib
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="HeartCare AI",
    page_icon="🫀",
    layout="centered"
)

# -------------------------------------------------
# CSS (CLEAN – NO HARD OVERRIDES)
# -------------------------------------------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background:
        linear-gradient(rgba(2,6,23,0.9), rgba(2,6,23,0.9)),
        url("https://images.unsplash.com/photo-1580281657521-6f9c3c58a6b1");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* HEADER CARD */
.header-card {
    background: linear-gradient(90deg, #0f172a, #1e40af, #0ea5e9);
    border-radius: 22px;
    padding: 28px 24px;
    margin: 25px 0 35px 0;
    box-shadow: 0 22px 50px rgba(14,165,233,0.45);
}

.header-card h1 {
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 6px;
}

.header-card p {
    text-align: center;
    font-size: 15px;
    color: #bae6fd;
}

/* FORM CARD */
.form-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 28px 65px rgba(0,0,0,0.45);
}

/* TITLE */
.form-title {
    font-size: 20px;
    font-weight: 800;
    color: #e5f0ff;
    margin-bottom: 22px;
    border-left: 5px solid #38bdf8;
    padding-left: 14px;
}

/* LABELS */
label {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
}
.stButton > button:hover {
    box-shadow: 0 16px 40px rgba(14,165,233,0.6);
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="header-card">
    <h1>🫀 HeartCare AI</h1>
    <p>Advanced clinical intelligence for early heart disease risk assessment</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# FORM
# -------------------------------------------------
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<div class="form-title">Patient Medical Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("Analyze Heart Health"):

    with st.spinner("AI analysis in progress..."):
        time.sleep(1.2)

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        scaled = scaler.transform(df)

        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100

    st.metric("Estimated Heart Disease Risk", f"{prob:.2f}%")
    st.progress(int(prob))

    if prediction == 1:
        st.error("⚠️ High risk detected. Please consult a medical professional.")
    else:
        st.success("✅ Low risk detected. Maintain a healthy lifestyle.")

# -------------------------------------------------
# FOOTER
st.markdown(
    """
    <p style='text-align:center; color:#94a3b8; margin-top:30px; font-size:16px; font-weight:600;'>
        HeartCare AI • AI-powered clinical decision support system
    </p>
    <p style='text-align:center; color:#64748b; margin-top:6px; font-size:15px;'>
        Machine Learning Project By <strong>Tejas Nikam</strong>
    </p>
    """,
    unsafe_allow_html=True
)

