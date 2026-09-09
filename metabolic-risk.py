import json
import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model



# Streamlit User Interface for Deployed Model
def main():
    st.title('Metabolic Risk Predictor')
    st.write('add your inputs to predict risk')
    
    # load model
    model = load_model("metaboli_risk_v0")
    
    # 1. Provide a text area for JSON/dictionary input
    json_input = st.text_area(
        "Enter feature values as a JSON dictionary:",
        '{"age": 28.0, "gender": 1.0, "ethnicity": 3.0, "poverty_ratio": 5.0, "education": 5.0, "marital_status": 1.0, "bmi": 27.700000762939453, "weight_kg": 85.69999694824219, "height_cm": 175.89999389648438, "waist_cm": 106.5, "hba1c": 5.300000190734863, "hdl_mgdl": 54.0, "total_chol": 200.0, "triglycerides": 88.0, "ldl_mgdl": 128.0, "crp_mgl": 2.0999999046325684, "vigorous_work_activity": 2.0, "moderate_work_activity": 1.0, "vigorous_minutes_per_day": null, "vigorous_recreational": 2.0, "moderate_rec_minutes": null, "calories": 1878.0, "carbs_g": 83.66999816894531, "fat_g": 107.37000274658203, "protein_g": 135.50999450683594, "sodium_mg": 3408.0, "saturated_fat_g": 30.531999588012695, "activity_score": 0.0, "protein_ratio": 0.28862619400024414}'
    )

    # 2. Parse and validate the input
    input_dict = None
    try:
      input_dict = json.loads(json_input)
      st.success("Valid dictionary format!")
      st.json(input_dict)  # Visual confirmation
    except json.JSONDecodeError as e:
      st.error(f"Invalid JSON format: {e}")

    # 3. Make predictions if input is valid
    if st.button("Predict") and input_dict:
        # Pass input_dict directly to your model or wrap it in a list/DataFrame
        input_df = pd.DataFrame([input_dict])
        # 3. CRUCIAL FIX: Force proper conversion of types
        # Convert Python None/null structures explicitly to numeric NaN or string 'missing'
        for col in input_df.columns:
            if input_df[col].dtype == "object":
            # If the column is categorical/text, fill nulls with a string placeholder or empty string
            input_df[col] = input_df[col].fillna("missing").astype(str)
            else:
            # If the column is numeric (float/int), enforce float type and proper np.nan
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        predictions = model.predict(input_df)
        st.write("Running prediction...") 
        # Show result
        st.success(f'Estimated price: ${predictions["prediction_score_1"]:,.2f}')
 
if __name__ == '__main__':
    main()
