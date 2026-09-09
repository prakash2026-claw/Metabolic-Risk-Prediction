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
        # 2. Load the template row and immediately clean up columns
        # 2. Load the template row and drop ID/Target
        df_sample = pd.read_csv("your_training_data_sample.csv", nrows=2)  # Read 2 rows now
        cols_to_drop = ["id", "target"]  # Update to match your exact dropped columns
        template_df = df_sample.drop(columns=cols_to_drop, errors="ignore").copy()

        clean_dtypes = template_df.dtypes.to_dict()

        # 3. Create the input dataframe
        # Row 0: User Input (Cleared out first)
        # Row 1: A valid reference dummy row (left completely intact with non-null training values)
        input_df = template_df.iloc[[0, 1]].copy().reset_index(drop=True)

        for col in input_df.columns:
          input_df.at[0, col] = np.nan  # Blank out ONLY the user row

        # 4. Overwrite Row 0 with user input dictionary
        # input_dict comes from your json.loads(json_input)
        for key, value in input_dict.items():
          if key in input_df.columns:
            input_df.at[0, key] = np.nan if value is None else value

        # 5. Enforce accurate data types across both rows
        input_df = input_df.astype(clean_dtypes)

        predictions = model.predict(input_df)
        st.write("Running prediction...") 
        # Show result
        st.success(f'Estimated price: ${predictions["prediction_score_1"]:,.2f}')
 
if __name__ == '__main__':
    main()
