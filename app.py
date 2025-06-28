import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📂 Churn Prediction from Excel or CSV")

# Load the trained model and scaler
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully.")
        st.subheader("📋 Input Data Preview:")
        st.dataframe(input_df)

        # Drop unnecessary columns if present
        columns_to_drop = ['UID', 'Target_ChurnFlag']
        input_df.drop(columns=[col for col in columns_to_drop if col in input_df.columns], inplace=True)

        # Handle 'X0' if it exists and is categorical
        if 'X0' in input_df.columns and input_df['X0'].dtype == 'object':
            input_df['X0'] = input_df['X0'].str.extract('(\d+)').astype(float)

        # Drop columns that are completely empty
        input_df = input_df.dropna(axis=1, how='all')

        # Check for missing or extra columns
        expected_cols = list(scaler.feature_names_in_)
        missing_cols = set(expected_cols) - set(input_df.columns)
        extra_cols = set(input_df.columns) - set(expected_cols)

        if missing_cols:
            st.error(f"❌ Missing required columns: {sorted(missing_cols)}")
        else:
            # Keep only required columns (silently ignore extra columns)
            input_df = input_df[expected_cols]

            # Scale and Predict
            input_scaled = scaler.transform(input_df)
            predictions = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)[:, 1]

            # Display results
            result_df = input_df.copy()
            result_df["Prediction"] = ["Churn" if p == 1 else "Not Churn" for p in predictions]
            result_df["Churn Probability"] = probabilities.round(4)

            st.subheader("📊 Prediction Results")
            st.dataframe(result_df)

            # 📈 Churn Distribution Bar Chart
            st.subheader("🔍 Churn Prediction Summary")
            churn_counts = result_df["Prediction"].value_counts()
            st.bar_chart(churn_counts)

            # 📌 Feature Importance (Optional)
            if st.checkbox("Show Top 10 Features Influencing Churn"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:]
                top_features = input_df.columns[indices]

                plt.figure(figsize=(8, 6))
                plt.barh(top_features, importances[indices])
                plt.title("Top 10 Features Affecting Churn")
                plt.xlabel("Importance")
                plt.tight_layout()
                st.pyplot(plt)

            # Downloadable CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")