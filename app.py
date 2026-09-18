import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the saved model
model_bundle = joblib.load("property_price_model.joblib")

model = model_bundle["model"]
dataset_start_date = pd.Timestamp(
    model_bundle["dataset_start_date"]
)


# Application title
st.title("Sydney Property Price Predictor")

st.write(
    "Enter the property information below to receive "
    "an estimated sale price."
)

st.info(
    "This application uses data from Liverpool, "
    "Burwood and Chatswood."
)


# Property input form
with st.form("property_form"):

    suburb = st.selectbox(
        "Suburb",
        ["Liverpool", "Burwood", "Chatswood"]
    )

    property_type = st.selectbox(
        "Property Type",
        ["House", "Apartment", "Unit", "Duplex", "Townhouse"]
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Number of Parking Spaces",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    land_size_unknown = st.checkbox(
        "Land size is unavailable"
    )

    if land_size_unknown:
        land_size = np.nan
    else:
        land_size = st.number_input(
            "Land Size (square metres)",
            min_value=0.0,
            value=400.0,
            step=10.0
        )

    internal_area_unknown = st.checkbox(
        "Internal area is unavailable"
    )

    if internal_area_unknown:
        internal_area = np.nan
    else:
        internal_area = st.number_input(
            "Internal Area (square metres)",
            min_value=0.0,
            value=90.0,
            step=5.0
        )

    sale_date = st.date_input(
        "Expected Sale Date"
    )

    predict_button = st.form_submit_button(
        "Predict Sale Price"
    )


# Generate the prediction
if predict_button:

    sale_date = pd.Timestamp(sale_date)

    # Create the engineered features
    total_rooms = bedrooms + bathrooms

    months_since_start = (
        (sale_date.year - dataset_start_date.year) * 12
        + sale_date.month
        - dataset_start_date.month
    )

    suburb_property_type = (
        suburb + "_" + property_type
    )

    land_size_missing = int(
        pd.isna(land_size)
    )

    internal_area_missing = int(
        pd.isna(internal_area)
    )

    # Create one property record
    input_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "parking": [parking],
        "land_size": [land_size],
        "internal_area": [internal_area],
        "total_rooms": [total_rooms],
        "months_since_start": [months_since_start],
        "land_size_missing": [land_size_missing],
        "internal_area_missing": [internal_area_missing],
        "suburb": [suburb],
        "property_type": [property_type],
        "suburb_property_type": [
            suburb_property_type
        ]
    })

    # Keep the same column order as the training data
    input_data = input_data[
        model_bundle["predictor_columns"]
    ]

    # Predict the sale price
    predicted_price = model.predict(
        input_data
    )[0]

    st.success(
        f"Estimated Sale Price: "
        f"AUD ${predicted_price:,.0f}"
    )

    st.warning(
        "This prediction is an estimate and should not "
        "replace a professional property valuation."
    )