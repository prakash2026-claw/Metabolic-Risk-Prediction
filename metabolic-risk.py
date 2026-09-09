import pandas as pd
from pycaret.classification import load_model, predict_model
import streamlit as st

# Set page configurations
st.set_page_config(page_title="Metabolic Risk Predictor", layout="centered")

# 1. Load the trained PyCaret model (cached so it only loads once)
@st.cache_resource
def get_model():
    # Make sure "metaboli_risk.pkl" is in the same directory as this script
    return load_model("metaboli_risk_v0")

model = get_model()

st.title("🏥 Metabolic Risk Prediction Dashboard")
st.write("Fill out the health and lifestyle indicators below to check the metabolic risk prediction.")

# 2. Build the Form Interface
with st.form("prediction_form"):
    st.subheader("Demographics & Physical Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=28.0)
        gender = st.number_input("Gender (Code)", min_value=0.0, value=1.0)
        ethnicity = st.number_input("Ethnicity (Code)", min_value=0.0, value=3.0)
    
    with col2:
        bmi = st.number_input("BMI", min_value=0.0, value=27.7)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=85.7)
        height_cm = st.number_input("Height (cm)", min_value=0.0, value=175.9)
        
    with col3:
        waist_cm = st.number_input("Waist Circumference (cm)", min_value=0.0, value=106.5)
        poverty_ratio = st.number_input("Poverty Ratio", min_value=0.0, value=5.0)
        education = st.number_input("Education Level (Code)", min_value=0.0, value=5.0)

    st.subheader("Lab Results & Blood Metrics")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        hba1c = st.number_input("HbA1c (%)", min_value=0.0, value=5.3)
        crp_mgl = st.number_input("CRP (mg/L)", min_value=0.0, value=2.1)
        total_chol = st.number_input("Total Cholesterol", min_value=0.0, value=200.0)
        
    with col5:
        hdl_mgdl = st.number_input("HDL (mg/dL)", min_value=0.0, value=54.0)
        ldl_mgdl = st.number_input("LDL (mg/dL)", min_value=0.0, value=128.0)
        
    with col6:
        triglycerides = st.number_input("Triglycerides", min_value=0.0, value=88.0)
        marital_status = st.number_input("Marital Status (Code)", min_value=0.0, value=1.0)

    st.subheader("Diet & Physical Activity")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        calories = st.number_input("Calories (kcal)", min_value=0.0, value=1878.0)
        carbs_g = st.number_input("Carbs (g)", min_value=0.0, value=83.7)
        fat_g = st.number_input("Fat (g)", min_value=0.0, value=107.4)
        
    with col8:
        protein_g = st.number_input("Protein (g)", min_value=0.0, value=135.5)
        saturated_fat_g = st.number_input("Saturated Fat (g)", min_value=0.0, value=30.5)
        sodium_mg = st.number_input("Sodium (mg)", min_value=0.0, value=3408.0)
        
    with col9:
        # Replaced raw 'nan' with None to cleanly show as empty or use custom defaults
        vigorous_minutes_per_day = st.number_input("Vigorous Minutes / Day", min_value=0.0, value=None, placeholder="Optional")
        moderate_rec_minutes = st.number_input("Moderate Rec Minutes", min_value=0.0, value=None, placeholder="Optional")
        activity_score = st.number_input("Activity Score", min_value=0.0, value=0.0)

    # Remaining activity variables
    col10, col11 = st.columns(2)
    with col10:
        vigorous_work_activity = st.number_input("Vigorous Work Activity (Code)", min_value=0.0, value=2.0)
        moderate_work_activity = st.number_input("Moderate Work Activity (Code)", min_value=0.0, value=1.0)
    with col11:
        vigorous_recreational = st.number_input("Vigorous Recreational (Code)", min_value=0.0, value=2.0)
        protein_ratio = st.number_input("Protein Ratio", min_value=0.0, value=0.29)

    # Submit button for the form
    submit_button = st.form_submit_button("Predict Metabolic Risk")

# 3. Handle Prediction Logic upon form submission
if submit_button:
    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {
        'age': age, 'gender': gender, 'ethnicity': ethnicity, 'poverty_ratio': poverty_ratio,
        'education': education, 'marital_status': marital_status, 'bmi': bmi, 'weight_kg': weight_kg,
        'height_cm': height_cm, 'waist_cm': waist_cm, 'hba1c': hba1c, 'hdl_mgdl': hdl_mgdl,
        'total_chol': total_chol, 'triglycerides': triglycerides, 'ldl_mgdl': ldl_mgdl, 'crp_mgl': crp_mgl,
        'vigorous_work_activity': vigorous_work_activity, 'moderate_work_activity': moderate_work_activity,
        'vigorous_minutes_per_day': vigorous_minutes_per_day, 'vigorous_recreational': vigorous_recreational,
        'moderate_rec_minutes': moderate_rec_minutes, 'calories': calories, 'carbs_g': carbs_g,
        'fat_g': fat_g, 'protein_g': protein_g, 'sodium_mg': sodium_mg, 'saturated_fat_g': saturated_fat_g,
        'activity_score': activity_score, 'protein_ratio': protein_ratio
    }
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])
    
    with st.spinner("Calculating risk..."):
        # Make the prediction using PyCaret
        predictions = predict_model(model, data=df)
        prediction_label = predictions["prediction_label"].iloc[0]
        
        # Display the result to the user
        st.success("### Prediction Complete!")
        st.metric(label="Risk Status Result", value=f"Class {prediction_label}")
