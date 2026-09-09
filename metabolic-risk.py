import streamlit as st
import pandas as pd
import numpy as np
from pydantic import create_model
from numpy import nan
from pycaret.classification import load_model, predict_model



# Streamlit User Interface for Deployed Model
def main():
    st.title('Metabolic Risk Predictor')
    st.write('add your inputs to predict risk')
    
    # load model
    model = load_model("metaboli_risk_v0")
    
    # Create input/output pydantic models
    input_model = create_model("metaboli_risk_input", **{'age': 28.0, 'gender': 1.0, 'ethnicity': 3.0, 'poverty_ratio': 5.0, 'education': 5.0, 'marital_status': 1.0, 'bmi': 27.700000762939453, 'weight_kg': 85.69999694824219, 'height_cm': 175.89999389648438, 'waist_cm': 106.5, 'hba1c': 5.300000190734863, 'hdl_mgdl': 54.0, 'total_chol': 200.0, 'triglycerides': 88.0, 'ldl_mgdl': 128.0, 'crp_mgl': 2.0999999046325684, 'vigorous_work_activity': 2.0, 'moderate_work_activity': 1.0, 'vigorous_minutes_per_day': nan, 'vigorous_recreational': 2.0, 'moderate_rec_minutes': nan, 'calories': 1878.0, 'carbs_g': 83.66999816894531, 'fat_g': 107.37000274658203, 'protein_g': 135.50999450683594, 'sodium_mg': 3408.0, 'saturated_fat_g': 30.531999588012695, 'activity_score': 0.0, 'protein_ratio': 0.28862619400024414})
    output_model = create_model("metaboli_risk_output", prediction=0)


    # 3. Make predictions if input is valid
    if st.button("Predict") and input_model:
        # Pass input_dict directly to your model or wrap it in a list/DataFrame
        input_df = pd.DataFrame([input_model.dict()])
        predictions = predict_model(model, data=data)
        st.write("Running prediction...") 
        # Show result
        st.success(f'Estimated price: ${predictions["prediction_score_1"]:,.2f}')
 
if __name__ == '__main__':
    main()
