import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Soft Shadows, Border-radius, and Cards
st.markdown("""
    <style>
    .main {
        background-color: #fdfbf7;
    }
    
    /* Modern card container with subtle shadow */
    .custom-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Prediction output card */
    .prediction-card {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
        margin: 15px 0px 25px 0px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_knn_model(model_path="knnRe.pkl"):
    """Load the pickled KNN regressor."""
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_knn_model("knnRe.pkl")

# Header section
st.title("🎓 Student Performance Predictor")
st.markdown("Enter student characteristics below to estimate the final performance score.")
st.write("---")

if model is None:
    st.error("⚠️ `knnRe.pkl` not found! Ensure the model file is placed in the exact same directory as this script.")
    st.stop()

# --- INPUT SECTION ---
# Sidebar Inputs
with st.sidebar:
    st.header("👤 Demographics & Program")
    age = st.slider("Age", min_value=15, max_value=60, value=21, step=1)
    
    gender_choice = st.selectbox("Gender", ["Female", "Male", "Other"])
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    gender = gender_map[gender_choice]
    
    course_choice = st.selectbox(
        "Course Stream", 
        ["Computer Science", "Engineering", "Business", "Medicine", "Arts", "Science"]
    )
    course_map = {
        "Computer Science": 0,
        "Engineering": 1,
        "Business": 2,
        "Medicine": 3,
        "Arts": 4,
        "Science": 5
    }
    course = course_map[course_choice]

    internet_access_choice = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
    internet_access = 1 if internet_access_choice == "Yes" else 0

# Main Page Inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📚 Study Routines")
    study_hours = st.slider("Weekly Study Hours", min_value=0.0, max_value=80.0, value=15.0, step=0.5)
    class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    
    study_method_choice = st.selectbox(
        "Primary Study Method",
        ["Self-study", "Group Study", "Online Lectures", "Tutoring / Coaching"]
    )
    study_method_map = {
        "Self-study": 0,
        "Group Study": 1,
        "Online Lectures": 2,
        "Tutoring / Coaching": 3
    }
    study_method = study_method_map[study_method_choice]
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🛌 Health & Exam Conditions")
    sleep_hours = st.slider("Daily Sleep Hours", min_value=2.0, max_value=12.0, value=7.0, step=0.5)
    sleep_quality = st.select_slider("Sleep Quality Rating (1=Poor, 5=Excellent)", options=[1, 2, 3, 4, 5], value=3)
    facility_rating = st.select_slider("Campus/Facility Rating (1=Poor, 5=Excellent)", options=[1, 2, 3, 4, 5], value=4)
    exam_difficulty = st.select_slider("Perceived Exam Difficulty (1=Easy, 5=Hard)", options=[1, 2, 3, 4, 5], value=3)
    st.markdown('</div>', unsafe_allow_html=True)

# Build DataFrame to match model schema
input_data = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'course': course,
    'study_hours': study_hours,
    'class_attendance': class_attendance,
    'internet_access': internet_access,
    'sleep_hours': sleep_hours,
    'sleep_quality': sleep_quality,
    'study_method': study_method,
    'facility_rating': facility_rating,
    'exam_difficulty': exam_difficulty
}])

# --- PREDICTION SECTION ---
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.subheader("🔮 Model Prediction")

pred_col1, pred_col2 = st.columns([1, 1])

with pred_col1:
    st.write("Click below to compute the model prediction for current inputs:")
    predict_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    auto_predict = st.checkbox("Auto-predict when values change", value=False)

with pred_col2:
    if predict_button or auto_predict:
        try:
            pred_score = model.predict(input_data)[0]
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="metric-label">Estimated Output / Score</div>
                    <div class="metric-value">{pred_score:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as err:
            st.error(f"Prediction failed: {err}")
    else:
        st.info("Press **🚀 Run Prediction** or toggle **Auto-predict** to generate a result.")

st.markdown('</div>', unsafe_allow_html=True)

# Feature preview table
with st.expander("🔍 View Processed Input Values"):
    st.dataframe(input_data, use_container_width=True)
