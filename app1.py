import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# PAGE CONFIG 
st.set_page_config(page_title="AI Agent – Women’s Health Dashboard", layout="wide")
st.title("🌸 AI Agent for Women’s Health")
st.write("An AI-powered preventive health assistant for women")

#  PRIVACY & MEMORY AGENT 
if 'history' not in st.session_state:
    st.session_state.history = []

#MODE SELECTION
mode = st.radio("Select Mode", ["Simple Mode", "Detailed Mode"], horizontal=True)

#  DATASET CREATION
def df(data):
    return pd.DataFrame(data)

repro_data = df({
    "age":[20,25,30,22,28,35,24,32],
    "irregular":[1,0,1,0,1,0,1,1],
    "weight_gain":[1,0,1,0,1,0,1,1],
    "acne":[1,0,1,0,1,0,1,1],
    "hair_growth":[1,0,1,0,1,0,1,1],
    "risk":[1,0,1,0,1,0,1,1]
})

anemia_data = df({
    "age":[18,22,25,30,35,40,28,19],
    "hb":[9.2,11.8,12.6,10.1,8.9,9.5,13.1,10.3],
    "fatigue":[8,3,2,6,9,7,1,6],
    "diet":[0,1,1,0,0,0,1,0],
    "flow":[1,0,0,1,1,1,0,1],
    "risk":[1,0,0,1,1,1,0,1]
})

preg_data = df({
    "age":[22,25,28,30,32,35,27,26],
    "bp":[130,120,140,135,128,122,138,125],
    "sugar":[90,85,110,105,95,88,115,100],
    "weight_gain":[1,0,1,0,1,0,1,1],
    "risk":[1,0,1,0,1,0,1,1]
})

lifestyle_data = df({
    "age":[25,30,28,35,40,32,27,29],
    "bmi":[25,22,30,28,35,23,31,26],
    "exercise":[0,1,0,1,0,1,0,1],
    "diet":[0,1,0,1,0,1,0,1],
    "risk":[1,0,1,0,1,0,1,1]
})

# MODEL TRAINING AGENT 
def train(df):
    X = df.drop("risk", axis=1)
    y = df["risk"]
    model = LogisticRegression()
    model.fit(X, y)
    return model

repro_model = train(repro_data)
anemia_model = train(anemia_data)
preg_model = train(preg_data)
life_model = train(lifestyle_data)

#  DATA COLLECTION AGENT 
st.header("📝 Daily Health Input")

age = st.number_input("Age", 15, 60, 25)

irregular = st.selectbox("Irregular Periods?", ["No", "Yes"])
weight_gain = st.selectbox("Weight Gain?", ["No", "Yes"])
acne = st.selectbox("Acne?", ["No", "Yes"])
hair_growth = st.selectbox("Excess Hair Growth?", ["No", "Yes"])

hb = st.number_input("Hemoglobin (g/dL)", 6.0, 15.0, 10.0)
fatigue = st.slider("Fatigue Level (0–10)", 0, 10, 5)
diet = st.selectbox("Diet Quality", ["Poor", "Good"])
flow = st.selectbox("Heavy Menstrual Flow?", ["No", "Yes"])

bp = st.number_input("Blood Pressure", 90, 180, 120)
sugar = st.number_input("Blood Sugar", 70, 200, 90)
preg_weight = st.selectbox("Pregnancy Weight Gain?", ["No", "Yes"])

bmi = st.number_input("BMI", 15, 40, 25)
exercise_min = st.number_input("Exercise Minutes", 0, 180, 30)

#  CONVERSION
bin_map = lambda x: 1 if x == "Yes" else 0
irregular, weight_gain, acne, hair_growth = map(bin_map, [irregular, weight_gain, acne, hair_growth])
diet, flow, preg_weight = map(lambda x: 1 if x == "Good" or x == "Yes" else 0, [diet, flow, preg_weight])
exercise = 1 if exercise_min >= 30 else 0

# ANALYSIS BUTTON
if st.button("🔍 Analyze My Health"):

    #  RISK PREDICTION AGENT 
    repro_risk = repro_model.predict_proba([[age, irregular, weight_gain, acne, hair_growth]])[0][1]
    anemia_risk = anemia_model.predict_proba([[age, hb, fatigue, diet, flow]])[0][1]
    preg_risk = preg_model.predict_proba([[age, bp, sugar, preg_weight]])[0][1]
    life_risk = life_model.predict_proba([[age, bmi, exercise, diet]])[0][1]

    risks = {
        "Reproductive Health": repro_risk,
        "Anemia": anemia_risk,
        "Pregnancy": preg_risk,
        "Lifestyle Disease": life_risk
    }

    score = int((1 - np.mean(list(risks.values()))) * 100)

    #  MEMORY AGENT 
    today = datetime.now().strftime("%Y-%m-%d")
    st.session_state.history.append({
        "date": today,
        "score": score,
        "exercise": exercise_min
    })
    st.session_state.history = st.session_state.history[-7:]

    #  REPORT
    st.subheader("🧠 Health Summary")
    st.metric("Overall Health Score", f"{score}/100")

    # EXPLAINABLE AI AGENT 
    st.subheader("📊 Risk Explanation")
    for k, v in risks.items():
        p = int(v * 100)
        if p < 30:
            st.success(f"{k}: {p}% (Low Risk)")
        elif p < 70:
            st.warning(f"{k}: {p}% (Moderate Risk)")
        else:
            st.error(f"{k}: {p}% (High Risk)")

    if hb < 10:
        st.info("🔍 Low hemoglobin is a major contributor to anemia risk.")
    if bp > 130 or sugar > 110:
        st.info("🔍 Elevated BP or sugar increases pregnancy-related risks.")
    if bmi > 30:
        st.info("🔍 High BMI contributes to lifestyle disease risk.")

    #  ESCALATION AGENT
    if max(risks.values()) > 0.75:
        st.error("🚨 Critical Risk Detected. Please consult a doctor immediately.")

    # WELLNESS COACH AGENT 
    st.subheader("🏋️ Wellness Coach")
    if exercise_min < 30:
        st.warning("Increase physical activity to at least 30 minutes daily.")
    else:
        st.success("Great job maintaining regular exercise!")

    # WEEKLY TREND 
    if mode == "Detailed Mode":
        st.subheader("📈 Weekly Health Trend")
        df_hist = pd.DataFrame(st.session_state.history).set_index("date")
        st.line_chart(df_hist["score"])
        st.bar_chart(df_hist["exercise"])

    st.info("⚠ AI provides guidance only. This is not a medical diagnosis.")
