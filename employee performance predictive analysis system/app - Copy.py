import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configurations
st.set_page_config(
    page_title="Predictive Employee Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-themed modern layout CSS
st.markdown("""
<style>
    /* Main background and global fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Center the title */
    .title-text {
        text-align: center;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        text-align: center;
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
    }
    
    /* Form containers and cards */
    .section-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* Prediction output card */
    .prediction-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        border: 2px solid #818cf8;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(129, 140, 248, 0.3);
    }
    
    .prediction-val {
        font-size: 4rem;
        font-weight: 900;
        color: #f43f5e;
        text-shadow: 0 0 10px rgba(244, 63, 94, 0.5);
        margin: 1rem 0;
    }
    
    /* Customize sliders and widgets */
    .stSlider > div > div > div > div {
        background-color: #818cf8;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #818cf8 100%);
        color: white;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.5);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(129, 140, 248, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.markdown("<h1 class='title-text'>Predictive Employee Performance Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Real-time Employee Performance Prediction powered by Random Forest Machine Learning</p>", unsafe_allow_html=True)

# Check model file exists
model_file = 'performance_model.pkl'
if not os.path.exists(model_file):
    st.error(f"⚠️ Model file '{model_file}' was not found. Please make sure the model is trained and saved in the same directory as this script.")
    st.info("💡 You can generate this model by running your Colab notebook and downloading the `performance_model.pkl` to this folder.")
    st.stop()

# Load model data
@st.cache_resource
def load_model_data():
    return joblib.load(model_file)

try:
    model_data = load_model_data()
    model = model_data['model']
    features = model_data['features']
    mappings = model_data['categorical_mappings']
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Sidebar Information Panel
with st.sidebar:
    st.markdown("### 📊 Project Insights")
    st.markdown("""
    This model utilizes a **Random Forest Classifier** trained on 100,000 employee records with an overall accuracy of **91.04%**.
    
    **Top Predictors:**
    - **Monthly Salary** (~73%)
    - **Job Title** (~16%)
    - **Satisfaction Score** (~1.3%)
    """)
    st.markdown("---")
    st.markdown("### 🔍 Model Properties")
    st.markdown(f"**Target Classes:** 5 (Low to High Performance)")
    st.markdown(f"**Random Forest Estimators:** 100")
    st.markdown(f"**Max Tree Depth:** 12")
    st.markdown("---")
    st.caption("Created for HR Operations and Talent Management optimization.")

# Create input form layout
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>👤 Employee Profile & Demographics</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    dept = st.selectbox("Department", options=mappings['Department'])
with col2:
    gender = st.selectbox("Gender", options=mappings['Gender'])
with col3:
    job_title = st.selectbox("Job Title", options=mappings['Job_Title'])
with col4:
    edu_level = st.selectbox("Education Level", options=mappings['Education_Level'])

col1_2, col2_2, col3_2 = st.columns(3)
with col1_2:
    age = st.slider("Age", min_value=18, max_value=65, value=35)
with col2_2:
    years_at_co = st.slider("Years At Company", min_value=0, max_value=45, value=5)
with col3_2:
    resigned_bool = st.selectbox("Resigned Status", options=["Active Employee", "Resigned"])
    resigned = 1 if resigned_bool == "Resigned" else 0

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>💼 Job Parameters & Compensation</div>", unsafe_allow_html=True)
col5, col6, col7, col8 = st.columns(4)

with col5:
    monthly_salary = st.number_input("Monthly Salary ($)", min_value=1000.0, max_value=30000.0, value=6500.0, step=100.0)
with col6:
    hours_per_week = st.slider("Work Hours Per Week", min_value=20, max_value=60, value=40)
with col7:
    remote_freq = st.selectbox("Remote Work Frequency (%)", options=[0, 50, 100], index=1)
with col8:
    team_size = st.slider("Team Size Managed/Assigned", min_value=1, max_value=30, value=10)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📈 Productivity & Engagement Metrics</div>", unsafe_allow_html=True)
col9, col10, col11, col12, col13 = st.columns(5)

with col9:
    projects = st.number_input("Projects Handled", min_value=0, max_value=100, value=8, step=1)
with col10:
    overtime_hrs = st.slider("Overtime Hours / Month", min_value=0, max_value=100, value=15)
with col11:
    sick_days = st.slider("Sick Days / Year", min_value=0, max_value=30, value=3)
with col12:
    training_hrs = st.slider("Training Hours / Year", min_value=0, max_value=100, value=25)
with col13:
    promotions = st.slider("Number of Promotions", min_value=0, max_value=10, value=0)

col_sat = st.columns([1, 2, 1])
with col_sat[1]:
    satisfaction = st.slider("Employee Satisfaction Score (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=3.4, step=0.1)

st.markdown("</div>", unsafe_allow_html=True)

# Encoding inputs and making prediction
# Encoders map labels to values. Let's find index of selected category.
dept_idx = mappings['Department'].index(dept)
gender_idx = mappings['Gender'].index(gender)
job_idx = mappings['Job_Title'].index(job_title)
edu_idx = mappings['Education_Level'].index(edu_level)

# Create input record in exact order of features
# Model features were: ['Department', 'Gender', 'Age', 'Job_Title', 'Years_At_Company', 'Education_Level', 'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency', 'Team_Size', 'Training_Hours', 'Promotions', 'Employee_Satisfaction_Score', 'Resigned']
input_data = pd.DataFrame([{
    'Department': dept_idx,
    'Gender': gender_idx,
    'Age': age,
    'Job_Title': job_idx,
    'Years_At_Company': years_at_co,
    'Education_Level': edu_idx,
    'Monthly_Salary': monthly_salary,
    'Work_Hours_Per_Week': hours_per_week,
    'Projects_Handled': projects,
    'Overtime_Hours': overtime_hrs,
    'Sick_Days': sick_days,
    'Remote_Work_Frequency': remote_freq,
    'Team_Size': team_size,
    'Training_Hours': training_hrs,
    'Promotions': promotions,
    'Employee_Satisfaction_Score': satisfaction,
    'Resigned': resigned
}])

# Reorder columns to match model's expected features
input_data = input_data[features]

st.markdown("---")

col_btn, col_out = st.columns([1, 1])

with col_btn:
    st.markdown("<div class='section-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center;'>", unsafe_allow_html=True)
    st.markdown("### Run Inference Engine")
    st.write("Submit the employee parameters to the Random Forest model. The engine computes the performance category distribution based on multi-dimensional patterns.")
    predict_btn = st.button("🔮 Calculate Performance Score")
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    if predict_btn:
        # Get predictions
        pred_class = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        
        # Color coding rating
        rating_colors = {
            1: "#ef4444",  # Red
            2: "#f97316",  # Orange
            3: "#eab308",  # Yellow
            4: "#10b981",  # Green
            5: "#06b6d4"   # Cyan
        }
        rating_labels = {
            1: "Needs Improvement (Low Performance)",
            2: "Below Expectations",
            3: "Meets Expectations (Average)",
            4: "Exceeds Expectations",
            5: "Outstanding Performance (Top Tier)"
        }
        
        color = rating_colors[pred_class]
        label = rating_labels[pred_class]
        
        st.markdown(f"""
        <div class='prediction-card'>
            <h3 style='color: #818cf8; margin: 0;'>Predicted Performance Rating</h3>
            <div class='prediction-val' style='color: {color}; text-shadow: 0 0 15px {color}33;'>{pred_class}</div>
            <h4 style='color: #f8fafc; margin: 0;'>{label}</h4>
            <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;'>Accuracy: ~91% | Confidence: {pred_proba[pred_class - 1]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plot probability distribution
        fig = go.Figure(go.Bar(
            x=['Rating 1', 'Rating 2', 'Rating 3', 'Rating 4', 'Rating 5'],
            y=pred_proba * 100,
            marker_color=[rating_colors[i] for i in range(1, 6)],
            text=[f"{p*100:.1f}%" for p in pred_proba],
            textposition='auto',
        ))
        fig.update_layout(
            title="Classification Confidence Distribution (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)', range=[0, 100]),
            xaxis=dict(gridcolor='rgba(0,0,0,0)'),
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class='prediction-card' style='border: 1px dashed rgba(148, 163, 184, 0.3); background: transparent;'>
            <h3 style='color: #94a3b8; margin: 2rem 0;'>Click 'Calculate Performance Score' to run prediction</h3>
        </div>
        """, unsafe_allow_html=True)
