import os
import pickle
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern layout and box-shadow styling
st.markdown("""
<style>
    /* Main background adjustment */
    .stApp {
        background-color: #0e1117;
        color: #f8fafc;
    }

    /* Top header styling */
    .hero-header {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
    }
    .hero-header h1 {
        font-size: 2.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .hero-header p {
        font-size: 1rem;
        color: #94a3b8;
    }

    /* Custom container cards with multi-layered drop shadows */
    .custom-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 
                    0 2px 4px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .custom-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 
                    0 4px 6px -4px rgba(0, 0, 0, 0.4);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Prediction Result Banner */
    .result-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(16, 185, 129, 0.15));
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.75rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 12px 20px -5px rgba(16, 185, 129, 0.25);
    }
    .result-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 800;
        color: #10b981;
        margin: 0.5rem 0;
    }

    /* Button shadow */
    div.stButton > button:first-child {
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338ca;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Load the model with caching
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "knnRe.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load `knnRe.pkl`: {e}")
    st.stop()

# Header
st.markdown("""
<div class="hero-header">
    <h1>Student Score Predictor</h1>
    <p>Estimate academic performance using your K-Nearest Neighbors Regression model</p>
</div>
""", unsafe_allow_html=True)

# Main Form organized into clean UI columns and cards
with st.form("prediction_form"):
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.markdown('<div class="custom-card"><div class="card-title">👤 Demographics & Course</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=10.0, max_value=100.0, value=20.0, step=1.0)
        gender = st.number_input("Gender (Encoded Value)", min_value=0.0, max_value=10.0, value=1.0, step=1.0, help="Numeric encoded gender (e.g. 0 or 1)")
        course = st.number_input("Course (Encoded Value)", min_value=0.0, max_value=50.0, value=2.0, step=1.0, help="Numeric identifier for enrolled program")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card"><div class="card-title">📚 Study & Attendance</div>', unsafe_allow_html=True)
        study_hours = st.number_input("Weekly Study Hours", min_value=0.0, max_value=120.0, value=15.0, step=0.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        study_method = st.number_input("Study Method (Encoded Value)", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
        internet_access = st.selectbox("Internet Access", options=[(1.0, "Yes (1)"), (0.0, "No (0)")], format_func=lambda x: x[1])[0]
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="custom-card"><div class="card-title">🌙 Lifestyle & Environment</div>', unsafe_allow_html=True)
        sleep_hours = st.number_input("Average Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        sleep_quality = st.slider("Sleep Quality Rating", min_value=1.0, max_value=10.0, value=7.0, step=1.0)
        facility_rating = st.slider("Facility Rating", min_value=1.0, max_value=10.0, value=8.0, step=1.0)
        exam_difficulty = st.slider("Exam Difficulty", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    submit_btn = st.form_submit_button("Predict Score", use_container_width=True)

# Inference and Display
if submit_btn:
    try:
        # Features ordered exactly as expected by the trained model:
        # ['age', 'gender', 'course', 'study_hours', 'class_attendance',
        #  'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
        #  'facility_rating', 'exam_difficulty']
        input_data = np.array([[
            float(age),
            float(gender),
            float(course),
            float(study_hours),
            float(class_attendance),
            float(internet_access),
            float(sleep_hours),
            float(sleep_quality),
            float(study_method),
            float(facility_rating),
            float(exam_difficulty)
        ]], dtype=np.float64)

        prediction = float(model.predict(input_data)[0])

        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">Predicted Performance Outcome</div>
            <div class="result-value">{prediction:.2f}</div>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Inference generated using KNeighborsRegressor</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Inference error: {err}")
