import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation (KMeans)", page_icon="🧠", layout="wide")


@st.cache_resource
def load_kmeans_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


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

# The saved KMeans has n_features_in_ = 7 (based on the notebook training).
if hasattr(model, "n_features_in_"):
    n = int(model.n_features_in_)
    feature_columns = feature_columns[:n]


st.title("🧠 Customer Segmentation")
st.caption("Predicts the cluster using the saved KMeans model from `kmeans_model2.pkl`.")

st.subheader("Input (manual) - numeric values")
st.info(
    "Training notebook standardized the features using StandardScaler before fitting KMeans. "
    "This app applies StandardScaler at prediction time."
)

cols = st.columns(2)
user_values = {}
for i, col in enumerate(feature_columns):
    with cols[i % 2]:
        user_values[col] = st.number_input(col, value=0.0, step=1.0)

input_df = pd.DataFrame([user_values]).astype(float)

st.subheader("Prediction")

if st.button("Predict cluster", type="primary"):
    try:
        # Match column order and notebook feature shape
        input_df = input_df[feature_columns]

        # Notebook did scaling before KMeans
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_df)

        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted Cluster: {int(pred)}")

        # For KMeans, `transform` gives distances to centroids
        if hasattr(model, "transform"):
            dists = model.transform(X_scaled)
            nearest = int(np.argmin(dists, axis=1)[0])
            st.write("Nearest cluster (min distance):", nearest)

    except Exception as e:
        st.error("Prediction failed. The model may expect a different feature shape/preprocessing.")
        st.exception(e)


with st.expander("Debug: show inferred feature columns"):
    st.write(feature_columns)


