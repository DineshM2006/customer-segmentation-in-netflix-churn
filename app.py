import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Segmentation (KMeans)", page_icon="🧠", layout="wide")


@st.cache_resource
def load_kmeans_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model2.pkl")
model = load_kmeans_model(MODEL_PATH)


# Feature columns from your `ML_projet_using_clusturing.ipynb`
# (the notebook builds X = df[features] with 7 columns)
feature_columns = [
    "account_age_months",
    "subscription_type",
    "country",
    "age",
    "monthly_fee",
    "primary_device",
    "devices_used",
]

# Your request said “best 10 columns”. This notebook uses top-7 features,
# so we keep exactly those 7.
if hasattr(model, "n_features_in_"):
    n = int(model.n_features_in_)
    feature_columns = feature_columns[:n]



st.title("🧠 Customer Segmentation")
st.caption("Loads your pre-trained KMeans model from `kmeans_model2.pkl` and performs manual predictions.")

st.subheader("Input (manual) - select your values")

st.info(
    "All inputs are treated as numeric values. If your notebook used encoding/scaling, "
    "make sure your saved model already contains the preprocessing pipeline.")

cols = st.columns(2)

user_values = {}
for i, col in enumerate(feature_columns):
    with cols[i % 2]:
        # numeric input - simple and robust for manual entry
        default_val = 0.0
        user_values[col] = st.number_input(f"{col}", value=float(default_val))

input_df = pd.DataFrame([user_values])

st.subheader("Prediction")

if st.button("Predict cluster", type="primary"):
    try:
        # KMeans returns label array
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Cluster: {int(pred)}")

        # If model supports probabilities (rare for KMeans), show more.
        if hasattr(model, "transform"):
            dists = model.transform(input_df)
            best = int(np.argmin(dists, axis=1)[0])
            st.write("Nearest cluster (min distance):", best)
    except Exception as e:
        st.error(
            "Prediction failed. This usually means the model expects different column names/dtypes "
            "than the ones provided."
        )
        st.exception(e)

# Developer hint
with st.expander("Debug: show inferred feature columns"):
    st.write(feature_columns)

