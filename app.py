import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime


st.set_page_config(page_title="Women’s Health AI Dashboard", layout="wide")
st.title("🌸 AI for Women’s Health Dashboard")
st.write("Track daily health, predict risks, log exercise, and get personalized advice.")


if 'history' not in st.session_state:
    st.session_state['history'] = []


def sample_df(data_dict):
    return pd.DataFrame(data_dict)


repro_data = sample_df({
    "age":[20,25,30,22,28,35,24,32],
    "irregular_periods":[1,0,1,0,1,0,1,1],
    "weight_gain":[1,0,1,0,1,0,1,1],
    "acne":[1,0,1,0,1,0,1,1],
    "hair_growth":[1,0,1,0,1,0,1,1],
    "risk":[1,0,1,0,1,0,1,1]
})
anemia_data = sample_df({
    "age":[18,22,25,30,35,40,28,19],
    "hemoglobin":[9.2,11.8,12.6,10.1,8.9,9.5,13.1,10.3],
    "fatigue":[8,3,2,6,9,7,1,6],
    "diet":[0,1,1,0,0,0,1,0],
    "heavy_flow":[1,0,0,1,1,1,0,1],
    "risk":[1,0,0,1,1,1,0,1]
})
preg_data = sample_df({
    "age":[22,25,28,30,32,35,27,26],
    "bp":[130,120,140,135,128,122,138,125],
    "sugar":[90,85,110,105,95,88,115,100],
    "weight_gain":[1,0,1,0,1,0,1,1],
    "risk":[1,0,1,0,1,0,1,1]
})
hormonal_data = sample_df({
    "age":[25,30,28,35,40,32,27,29],
    "fatigue":[6,3,7,2,8,4,6,5],
    "weight_change":[1,0,1,0,1,0,1,1],
    "hair_loss":[1,0,1,0,1,0,1,1],
    "risk":[1,0,1,0,1,0,1,1]
})
lifestyle_data = sample_df({
    "age":[25,30,28,35,40,32,27,29],
    "bmi":[25,22,30,28,35,23,31,26],
    "exercise":[0,1,0,1,0,1,0,1],
    "diet":[0,1,0,1,0,1,0,1],
    "risk":[1,0,1,0,1,0,1,1]
})
cancer_data = sample_df({
    "age":[30,35,40,32,28,45,37,33],
    "family_history":[1,0,1,0,0,1,1,0],
    "bp":[130,120,140,135,125,128,138,122],
    "weight":[70,60,80,75,68,85,72,78],
    "risk":[1,0,1,0,0,1,1,0]
})

# ------------------ TRAIN MODELS ------------------
def train_model(df):
    X = df.drop("risk", axis=1)
    y = df["risk"]
    model = LogisticRegression()
    model.fit(X, y)
    return model

repro_model = train_model(repro_data)
anemia_model = train_model(anemia_data)
preg_model = train_model(preg_data)
hormonal_model = train_model(hormonal_data)
lifestyle_model = train_model(lifestyle_data)
cancer_model = train_model(cancer_data)

# ------------------ USER INPUT ------------------
st.header("📝 Enter Your Daily Health Details")

# 1️⃣ Menstrual & Reproductive
st.subheader("1️⃣ Menstrual & Reproductive Problems")
col1, col2 = st.columns(2)
with col1:
    age_repro = st.number_input("Age", 15, 60, 25)
    irregular = st.selectbox("Irregular Periods?", ["No", "Yes"])
    weight_gain_repro = st.selectbox("Weight Gain?", ["No", "Yes"])
with col2:
    acne = st.selectbox("Acne Problem?", ["No", "Yes"])
    hair_growth = st.selectbox("Excess Hair Growth?", ["No", "Yes"])

# 2️⃣ Anemia & Nutrition
st.subheader("2️⃣ Anemia & Nutritional Deficiency")
col1, col2 = st.columns(2)
with col1:
    hemoglobin = st.number_input("Hemoglobin Level (g/dL)", 6.0, 15.0, 10.0)
    fatigue = st.slider("Fatigue Level (0-10)", 0, 10, 5)
with col2:
    diet_anemia = st.selectbox("Diet Quality", ["Poor", "Good"])
    heavy_flow = st.selectbox("Heavy Menstrual Flow?", ["No", "Yes"])

# 3️⃣ Pregnancy & Maternal
st.subheader("3️⃣ Pregnancy & Maternal Health")
col1, col2 = st.columns(2)
with col1:
    bp = st.number_input("Blood Pressure (mmHg)", 90, 180, 120)
    sugar = st.number_input("Blood Sugar (mg/dL)", 70, 200, 90)
with col2:
    weight_gain_preg = st.selectbox("Pregnancy Weight Gain?", ["No", "Yes"])

# 4️⃣ Hormonal Disorders
st.subheader("4️⃣ Hormonal Disorders")
col1, col2 = st.columns(2)
with col1:
    weight_change = st.selectbox("Recent Weight Change?", ["No", "Yes"])
with col2:
    hair_loss = st.selectbox("Hair Loss?", ["No", "Yes"])

# 5️⃣ Lifestyle & Exercise
st.subheader("5️⃣ Lifestyle & Exercise")
col1, col2 = st.columns(2)
with col1:
    bmi = st.number_input("BMI", 15, 40, 25)
    workout_minutes = st.number_input("Workout Duration (minutes)", 0, 180, 30)
with col2:
    exercise_type = st.selectbox("Type of Exercise", ["None", "Cardio", "Strength", "Yoga", "Mixed"])
    diet_life = st.selectbox("Diet Healthy?", ["No", "Yes"])

# Daily Checklist
st.markdown("**📋 Daily Health Checklist**")
st.checkbox("Drink 8 glasses of water")
st.checkbox("30 mins light exercise / walk")
st.checkbox("Eat iron-rich food (spinach, lentils)")
st.checkbox("Sleep 7-8 hours")
st.checkbox("Monitor BP & sugar (if applicable)")

# 6️⃣ Cancer
st.subheader("6️⃣ Cancer Risk")
col1, col2 = st.columns(2)
with col1:
    family_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])
    weight_cancer = st.number_input("Weight (kg)", 40, 120, 65)
with col2:
    st.write("💡 Cancer risk depends on family history, age, BP & weight")

# ------------------ CONVERT INPUTS ------------------
irregular = 1 if irregular=="Yes" else 0
weight_gain_repro = 1 if weight_gain_repro=="Yes" else 0
acne = 1 if acne=="Yes" else 0
hair_growth = 1 if hair_growth=="Yes" else 0
diet_anemia = 1 if diet_anemia=="Good" else 0
heavy_flow = 1 if heavy_flow=="Yes" else 0
weight_gain_preg = 1 if weight_gain_preg=="Yes" else 0
weight_change = 1 if weight_change=="Yes" else 0
hair_loss = 1 if hair_loss=="Yes" else 0
exercise_done = 1 if workout_minutes >= 30 else 0
diet_life = 1 if diet_life=="Yes" else 0
family_history = 1 if family_history=="Yes" else 0

# ------------------ PREDICTION & SAVE ------------------
if st.button("🔍 Analyze Health Today"):

    # Predict probabilities
    repro_prob = repro_model.predict_proba([[age_repro, irregular, weight_gain_repro, acne, hair_growth]])[0][1]
    anemia_prob = anemia_model.predict_proba([[age_repro, hemoglobin, fatigue, diet_anemia, heavy_flow]])[0][1]
    preg_prob = preg_model.predict_proba([[age_repro, bp, sugar, weight_gain_preg]])[0][1]
    hormonal_prob = hormonal_model.predict_proba([[age_repro, fatigue, weight_change, hair_loss]])[0][1]
    lifestyle_prob = lifestyle_model.predict_proba([[age_repro, bmi, exercise_done, diet_life]])[0][1]
    cancer_prob = cancer_model.predict_proba([[age_repro, family_history, bp, weight_cancer]])[0][1]

    risks = {
        "Menstrual/Reproductive": repro_prob,
        "Anemia/Nutrition": anemia_prob,
        "Pregnancy/Maternal": preg_prob,
        "Hormonal Disorder": hormonal_prob,
        "Lifestyle Disease": lifestyle_prob,
        "Cancer": cancer_prob
    }

    # Overall Health Score
    overall_score = int((1 - np.mean(list(risks.values())))*100)

    # Save today's data
    today = datetime.now().strftime("%Y-%m-%d")
    st.session_state['history'].append({
        'date': today,
        'Overall Health': overall_score,
        'Workout (min)': workout_minutes,
        **{k: int(v*100) for k,v in risks.items()}
    })
    if len(st.session_state['history'])>7:
        st.session_state['history'] = st.session_state['history'][-7:]

    # ------------------ TODAY'S REPORT ------------------
    st.subheader("🧠 Today's Health Report")
    st.metric("💖 Overall Health Score", f"{overall_score}/100")

    for condition, prob in risks.items():
        percent = int(prob*100)
        if percent < 30: st.success(f"{condition}: {percent}% risk – Low")
        elif percent < 70: st.warning(f"{condition}: {percent}% risk – Moderate")
        else: st.error(f"{condition}: {percent}% risk – High")

    # Exercise Feedback
    st.subheader("🏋️ Daily Exercise Feedback")
    if workout_minutes < 30:
        st.warning(f"⚠ You exercised only {workout_minutes} minutes today. Aim for at least 30 mins. Suggested: Cardio or Strength training.")
    else:
        st.success(f"✅ Great! You exercised {workout_minutes} mins today ({exercise_type}). Keep it up!")

    # Personalized Recommendations
    st.subheader("💡 Recommendations Based on Risk")
    recommendations = {
        "Menstrual/Reproductive": "Consult gynecologist; maintain balanced diet & exercise.",
        "Anemia/Nutrition": "Increase iron-rich foods; consult doctor if symptoms persist.",
        "Pregnancy/Maternal": "Regular checkups; monitor BP & sugar.",
        "Hormonal Disorder": "Get hormonal tests; maintain healthy lifestyle.",
        "Lifestyle Disease": "Exercise regularly; balanced diet.",
        "Cancer": "Consult specialist; regular screening."
    }
    for cond, prob in risks.items():
        if prob > 0.5:
            st.warning(f"{cond}: {int(prob*100)}% risk. {recommendations[cond]}")

    st.info("📢 AI generated advice is informational only; consult a doctor for medical advice.")
