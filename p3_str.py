import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="💰"
)


st.title("🔮 Predict YouTube Ad Revenue")

MODEL_PATH = 'Huber_regression_model.pkl'

if not os.path.exists(MODEL_PATH):
    st.error(" Model file not found. Please check the path.")
    st.stop()

model = joblib.load(MODEL_PATH)

feature_cols = [
    'views',
    'likes',
    'comments',
    'watch_time_minutes',
    'video_length_minutes',
    'subscribers'
]

input_data = {}
cols_per_row = 3
rows = (len(feature_cols) + cols_per_row - 1) // cols_per_row

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i * cols_per_row + j
        if idx < len(feature_cols):
            col_name = feature_cols[idx]
            input_data[col_name] = cols[j].number_input(f"{col_name}", value=0.0)

available_categories = [
    'Entertainment',
    'Lifestyle',
    'Gaming',
    'Music',
    'Tech'
]

# Use st.selectbox for category selection
selected_category = st.selectbox(
    "Select Video Category",
    options=available_categories,
    index=0 # Default to 'Entertainment'
)

# Map selected category to one-hot encoding
category_one_hot = {
    f"category_{cat}": (1 if cat == selected_category else 0)
    for cat in available_categories
}
available_countries = [
    'US',
    'UK',
    'IN',
    'CA',
    'DE'
]

# Use st.selectbox for country selection
selected_country = st.selectbox(
    "Select Country",
    options=available_countries,
    index=0 # Default to 'US'
)

# Map selected country to one-hot encoding
country_one_hot = {
    f"country_{country}": (1 if country == selected_country else 0)
    for country in available_countries
}

day = st.number_input(
    "Day of Month (1-31)",
    min_value=1,
    max_value=31,
    value=15,
    step=1
)
day_of_week_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_of_week_numeric_options = list(range(7)) # 0 for Monday, 6 for Sunday
selected_day_of_week_index = st.selectbox(
    "Day of Week",
    options=day_of_week_numeric_options,
    format_func=lambda x: day_of_week_options[x],
    index=0 # Default to Monday (index 0)
)


device_options = ['Mobile', 'TV', 'Tablet']
selected_device = st.selectbox(
    "Device Type",
    options=device_options,
    index=0 # Default to Mobile
)
device_one_hot = {
    f"device_{device}": (1 if device == selected_device else 0)
    for device in device_options
}
engagement_rate= (input_data['likes'] + input_data['comments']) / max(input_data['views'], 1)
year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2050,
    value=2025,
    step=1
)

month = st.number_input(
    "Month (1-12)",
    min_value=1,
    max_value=12,
    value=7,
    step=1
)






varied_inputs = {
    "year": year ,
    "month": month,
    "day": day,
    "day_of_week": selected_day_of_week_index,
    "engagement_rate": engagement_rate,
    **device_one_hot,  
    **category_one_hot,
    **country_one_hot     
}

combined_input_data = {**input_data, **varied_inputs}

MODEL_EXPECTED_FEATURES = [
    'views',
    'likes',
    'comments',
    'watch_time_minutes',
    'video_length_minutes',
    'subscribers',
    'year',
    'month',
    'day',  
    'day_of_week',
    'engagement_rate',
    'category_Entertainment',
    'category_Gaming',
    'category_Lifestyle',
    'category_Music',
    'category_Tech',
    'device_Mobile',
    'device_TV',
    'device_Tablet',
    'country_CA',
    'country_DE',
    'country_IN',
    'country_UK',
    'country_US',
]




if st.button("Predict"):
    # Create the DataFrame with the correct column order
    input_df = pd.DataFrame([combined_input_data], columns=MODEL_EXPECTED_FEATURES)


    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)  
        st.success(f"💰 **Predicted Ad Revenue:** ${prediction/10000}")

        if prediction == 0:
            st.info("This video may not generate ad revenue based on the current input values.")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")