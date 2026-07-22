"""
Smart Classroom Management System — Streamlit App
Loads the trained model artifacts (.pkl) and provides:
  1. Single-student dropout risk prediction with recommended action
  2. Batch prediction from an uploaded CSV
  3. A dataset overview dashboard
Run with:  streamlit run app.py
Make sure these files are in the SAME folder as app.py:
  - dropout_risk_model.pkl
  - feature_scaler.pkl
  - classroom_segment_model.pkl
  - cluster_scaler.pkl
  - model_metadata.pkl
  - Educational_Management_Dataset.csv   (optional, only needed for the Dashboard tab)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Classroom Management System",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("dropout_risk_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    segment_model = joblib.load("classroom_segment_model.pkl")
    cluster_scaler = joblib.load("cluster_scaler.pkl")
    metadata = joblib.load("model_metadata.pkl")
    return model, scaler, segment_model, cluster_scaler, metadata

try:
    model, scaler, segment_model, cluster_scaler, metadata = load_artifacts()
    feature_cols = metadata["feature_cols"]
    cluster_features = metadata["cluster_features"]
    segment_labels_map = metadata["segment_labels_map"]
    best_model_name = metadata["best_model_name"]
    ARTIFACTS_LOADED = True
except Exception as e:
    ARTIFACTS_LOADED = False
    LOAD_ERROR = str(e)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def predict_risk(student_dict):
    x = pd.DataFrame([student_dict])[feature_cols]
    x_scaled = scaler.transform(x)
    proba = model.predict_proba(x_scaled)[0, 1]
    pred = int(proba >= 0.5)
    return proba, pred


def get_segment(student_dict):
    x_cluster = pd.DataFrame([student_dict])[cluster_features]
    x_scaled = cluster_scaler.transform(x_cluster)
    cluster_id = segment_model.predict(x_scaled)[0]
    return segment_labels_map.get(cluster_id, f"Segment {cluster_id}")


def risk_action(proba):
    if proba >= 0.7:
        return "🔴 High Risk", "Recommend immediate advisor meeting and academic support plan.", "red"
    elif proba >= 0.4:
        return "🟠 Moderate Risk", "Recommend a check-in and monitor engagement over the next 2 weeks.", "orange"
    else:
        return "🟢 Low Risk", "Student appears on track. Continue routine monitoring.", "green"


def batch_predict(df_batch):
    missing = [c for c in feature_cols if c not in df_batch.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")

    x = df_batch[feature_cols]
    x_scaled = scaler.transform(x)
    proba = model.predict_proba(x_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df_batch.copy()
    out["Dropout_Probability"] = np.round(proba, 3)
    out["Predicted_At_Risk"] = pred

    if all(c in df_batch.columns for c in cluster_features):
        x_cluster = cluster_scaler.transform(df_batch[cluster_features])
        cluster_ids = segment_model.predict(x_cluster)
        out["Classroom_Segment"] = [segment_labels_map.get(c, f"Segment {c}") for c in cluster_ids]

    return out


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🎓 Smart Classroom")
page = st.sidebar.radio(
    "Navigate",
    ["Single Student Prediction", "Batch Prediction (CSV)", "Dataset Dashboard"]
)

if ARTIFACTS_LOADED:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Active model: **{best_model_name}**")
else:
    st.sidebar.error("Model artifacts not found. See main panel for details.")

# ----------------------------------------------------------------------------
# Main title
# ----------------------------------------------------------------------------
st.title("🎓 Smart Classroom Management System")
st.caption("AI-powered dropout risk prediction & classroom engagement segmentation")

if not ARTIFACTS_LOADED:
    st.error(
        "Could not load model files. Make sure the following are in the same folder as app.py:\n\n"
        "- dropout_risk_model.pkl\n"
        "- feature_scaler.pkl\n"
        "- classroom_segment_model.pkl\n"
        "- cluster_scaler.pkl\n"
        "- model_metadata.pkl\n\n"
        f"Error details: {LOAD_ERROR}"
    )
    st.stop()

# ----------------------------------------------------------------------------
# PAGE 1: Single Student Prediction
# ----------------------------------------------------------------------------
if page == "Single Student Prediction":
    st.header("🧑‍🎓 Single Student Risk Assessment")
    st.write("Enter a student's academic and engagement details to get an instant risk prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=40, value=21)
        gpa = st.number_input("GPA (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0)
        assignment_avg = st.slider("Assignment Average (%)", 0.0, 100.0, 65.0)

    with col2:
        exam_score = st.slider("Exam Score (%)", 0.0, 100.0, 60.0)
        lms_login = st.number_input("LMS Login Frequency (per month)", min_value=0, max_value=100, value=15)
        video_time = st.number_input("Video Engagement Time (hrs/week)", min_value=0.0, max_value=50.0, value=8.0, step=0.1)

    with col3:
        forum = st.number_input("Forum Interactions", min_value=0, max_value=100, value=10)
        engagement_score = st.slider("Engagement Score (0-1)", 0.0, 1.0, 0.6)
        stress_index = st.slider("Stress Level Index (0-1)", 0.0, 1.0, 0.5)

    if st.button("🔍 Predict Dropout Risk", type="primary"):
        student = {
            "Age": age,
            "GPA": gpa,
            "Attendance_Rate": attendance,
            "Assignment_Avg": assignment_avg,
            "Exam_Score": exam_score,
            "LMS_Login_Frequency": lms_login,
            "Video_Engagement_Time": video_time,
            "Forum_Interactions": forum,
            "Engagement_Score": engagement_score,
            "Stress_Level_Index": stress_index,
        }

        proba, pred = predict_risk(student)
        label, action, color = risk_action(proba)
        segment = get_segment(student)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Dropout Probability", f"{proba*100:.1f}%")
        r2.metric("Risk Level", label)
        r3.metric("Classroom Segment", segment)

        st.progress(min(int(proba * 100), 100))

        if color == "red":
            st.error(f"**Recommended Action:** {action}")
        elif color == "orange":
            st.warning(f"**Recommended Action:** {action}")
        else:
            st.success(f"**Recommended Action:** {action}")

# ----------------------------------------------------------------------------
# PAGE 2: Batch Prediction
# ----------------------------------------------------------------------------
elif page == "Batch Prediction (CSV)":
    st.header("📂 Batch Prediction from CSV")
    st.write(
        "Upload a CSV with the same feature columns as the training dataset "
        f"({', '.join(feature_cols)}) to score multiple students at once."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())

            if st.button("▶️ Run Batch Prediction", type="primary"):
                result = batch_predict(df_batch)
                st.success(f"Scored {len(result)} students.")
                st.dataframe(result)

                st.subheader("Risk Distribution")
                st.bar_chart(result["Predicted_At_Risk"].value_counts())

                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_out,
                    file_name="dropout_risk_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ----------------------------------------------------------------------------
# PAGE 3: Dataset Dashboard
# ----------------------------------------------------------------------------
elif page == "Dataset Dashboard":
    st.header("📊 Dataset Overview")

    DATA_FILE = "Educational_Management_Dataset.csv"
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", len(df))
        c2.metric("At-Risk Students", int(df["Dropout_Risk"].sum()))
        c3.metric("Dropout Rate", f"{df['Dropout_Risk'].mean()*100:.1f}%")
        c4.metric("Avg GPA", f"{df['GPA'].mean():.2f}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dropout Risk Distribution")
            st.bar_chart(df["Dropout_Risk"].value_counts())

        with col2:
            st.subheader("Attendance Rate Distribution")
            st.bar_chart(np.histogram(df["Attendance_Rate"], bins=10)[0])

        st.subheader("Feature Correlation with Dropout Risk")
        num_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c != "Dropout_Risk"]
        corr = df[num_cols + ["Dropout_Risk"]].corr()["Dropout_Risk"].drop("Dropout_Risk").sort_values()
        st.bar_chart(corr)

        st.subheader("Raw Data Sample")
        st.dataframe(df.head(20))
    else:
        st.warning(
            f"'{DATA_FILE}' not found in the app folder. "
            "Place the dataset CSV next to app.py to see the dashboard."
        )
