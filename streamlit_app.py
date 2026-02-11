import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import base64

# ---------------------------
# PAGE
# ---------------------------
st.set_page_config(page_title="Hospital No-Show Prediction", layout="wide")

# ---------------------------
# MODEL
# ---------------------------
model = joblib.load("final_random_forest_model.pkl")

# ---------------------------
# BACKGROUND
# ---------------------------
def get_base64_bg(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_bg("bg.jpg")

# ---------------------------
# CSS (FINAL CLEAN)
# ---------------------------
st.markdown(f"""
<style>

/* ============================= */
/* BACKGROUND */
/* ============================= */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* hide header/footer */
[data-testid="stHeader"], footer {{
    display:none;
}}

/* ============================= */
/* SIDEBAR */
/* ============================= */
section[data-testid="stSidebar"] {{
    background: #0e1117;
}}

section[data-testid="stSidebar"] * {{
    color:white !important;
}}

/* ============================= */
/* REMOVE RANDOM SPACING / BARS */
/* ============================= */
.element-container:empty {{
    display:none !important;
}}

div[data-testid="stMarkdownContainer"]:empty {{
    display:none !important;
}}

hr {{
    display:none !important;
}}

/* tighten top/bottom */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}}

/* ============================= */
/* TEXT VISIBILITY */
/* ============================= */
h1, h2, h3, h4, h5, h6, p, span, label {{
    color: white !important;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.7);
}}

/* ============================= */
/* GLASS CARDS */
/* ============================= */
.card {{
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(6px);
    padding: 22px;
    border-radius: 18px;
    margin: 18px 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}}

/* ============================= */
/* SUMMARY LINE */
/* ============================= */
.summary-line {{
    background: rgba(255,255,255,0.15);
    padding: 12px;
    border-radius: 12px;
    font-weight: 600;
}}

/* ============================= */
/* BARS */
/* ============================= */
.risk-track {{
    height: 14px;
    background: rgba(255,255,255,0.25);
    border-radius: 20px;
    overflow: hidden;
    margin-top: 6px;
}}

.risk-fill {{
    height: 100%;
    background: linear-gradient(90deg,#22c55e,#eab308,#ef4444);
}}

.conf-track {{
    height: 10px;
    background: rgba(255,255,255,0.25);
    border-radius: 20px;
    overflow: hidden;
    margin-top: 6px;
}}

.conf-fill {{
    height: 100%;
    background: #3b82f6;
}}

/* ============================= */
/* BUTTON */
/* ============================= */
.stButton>button {{
    width:100%;
    height:3em;
    border-radius:12px;
    background:#2563eb;
    color:white;
    font-weight:600;
    border:none;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("Patient Information")

age = st.sidebar.slider("Age", 0, 100, 30)
sms = st.sidebar.selectbox("SMS Reminder", ["Yes", "No"])
hypertension = st.sidebar.selectbox("Hypertension", ["Yes", "No"])
diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"])
alcoholism = st.sidebar.selectbox("Alcoholism", ["Yes", "No"])
handicap = st.sidebar.selectbox("Handicap", ["Yes", "No"])
scholarship = st.sidebar.selectbox("Scholarship", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5)

# ---------------------------
# ENCODING
# ---------------------------
def encode_binary(v):
    return 1 if v == "Yes" else 0

def create_input_dataframe():
    data = {
        "age": age,
        "sms_received": encode_binary(sms),
        "hypertension": encode_binary(hypertension),
        "diabetes": encode_binary(diabetes),
        "alcoholism": encode_binary(alcoholism),
        "handicap": encode_binary(handicap),
        "scholarship": encode_binary(scholarship),
        "gender": 1 if gender == "Male" else 0,
    }

    df = pd.DataFrame([data])

    df["age_group_19-30"] = 1 if 19 <= age <= 30 else 0
    df["age_group_31-45"] = 1 if 31 <= age <= 45 else 0
    df["age_group_46-60"] = 1 if 46 <= age <= 60 else 0
    df["age_group_61-75"] = 1 if 61 <= age <= 75 else 0
    df["age_group_76+"] = 1 if age >= 76 else 0

    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df

# ---------------------------
# TITLE
# ---------------------------
st.title("Hospital Appointment No-Show Prediction")
st.write("AI-powered risk evaluation for patient attendance.")

# ---------------------------
# PATIENT SUMMARY
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patient Summary")

summary_text = (
    f"Age: {age} | "
    f"Gender: {gender} | "
    f"SMS: {sms} | "
    f"Hypertension: {hypertension} | "
    f"Diabetes: {diabetes} | "
    f"Alcoholism: {alcoholism} | "
    f"Handicap: {handicap} | "
    f"Scholarship: {scholarship} | "
    f"Threshold: {threshold:.2f}"
)

st.markdown(f'<div class="summary-line">{summary_text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("Predict Appointment Outcome"):

    input_df = create_input_dataframe()
    prob = model.predict_proba(input_df)[0][1]
    prediction = "No-Show" if prob >= threshold else "Attend"

    # ---------------------------
    # Clinical
    # ---------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Clinical Decision Panel")
    st.write(f"### Predicted: **{prediction}**")
    st.write(f"Probability of No-Show: **{prob*100:.2f}%**")

    st.markdown(f"""
    <div class="risk-track">
        <div class="risk-fill" style="width:{prob*100:.2f}%"></div>
    </div>
    """, unsafe_allow_html=True)

    confidence = min(1.0, abs(prob-threshold)/0.5)
    st.write(f"Confidence: **{confidence*100:.1f}%**")

    st.markdown(f"""
    <div class="conf-track">
        <div class="conf-fill" style="width:{confidence*100:.2f}%"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Performance
    # ---------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "0.80")
    c2.metric("Recall", "0.51")
    c3.metric("F1 Score", "0.64")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Feature importance
    # ---------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Importance")

    importances = model.feature_importances_
    features = model.feature_names_in_

    feat_df = pd.DataFrame({"Feature": features, "Importance": importances}) \
        .sort_values(by="Importance").tail(10)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(feat_df["Feature"], feat_df["Importance"])
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
