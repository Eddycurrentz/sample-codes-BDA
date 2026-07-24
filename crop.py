import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


def build_sample_dataset():
    """Create a simple synthetic dataset for crop monitoring demo purposes."""
    np.random.seed(42)
    n = 120

    data = pd.DataFrame(
        {
            "ndvi": np.random.uniform(0.2, 0.9, n),
            "soil_moisture": np.random.uniform(0.1, 0.8, n),
            "temperature": np.random.uniform(20, 35, n),
            "precipitation": np.random.uniform(0, 120, n),
        }
    )

    data["growth_stage_name"] = np.random.choice(
        ["vegetative", "flowering", "maturity"],
        size=n,
    )

    data["crop_status"] = data.apply(
        lambda row: (
            "healthy"
            if row["ndvi"] > 0.7 and row["soil_moisture"] > 0.5
            else "stress"
            if row["growth_stage_name"] == "flowering" and row["temperature"] > 30
            else "moderate"
        ),
        axis=1,
    )

    encoder = LabelEncoder()
    data["growth_stage"] = encoder.fit_transform(data["growth_stage_name"])
    return data, encoder


def train_crop_model(data, encoder, model_type="random_forest"):
    features = ["ndvi", "soil_moisture", "temperature", "precipitation", "growth_stage"]
    X = data[features]
    y = data["crop_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    if model_type == "xgboost" and xgb is not None:
        model = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print("Model accuracy:", round(accuracy, 3))
    print(classification_report(y_test, preds))
    return model


def predict_crop_status(model, encoder, features):
    feature_df = pd.DataFrame([features])
    feature_df["growth_stage"] = encoder.transform([features["growth_stage"]])[0]
    feature_df = feature_df[["ndvi", "soil_moisture", "temperature", "precipitation", "growth_stage"]]
    return model.predict(feature_df)[0]


def run_demo():
    print("Loading sample crop monitoring dataset...")
    data, encoder = build_sample_dataset()
    print("Training crop status model...")
    model = train_crop_model(data, encoder)

    sample_input = {
        "ndvi": 0.82,
        "soil_moisture": 0.65,
        "temperature": 28,
        "precipitation": 40,
        "growth_stage": "flowering",
    }

    prediction = predict_crop_status(model, encoder, sample_input)
    print("Predicted crop status:", prediction)


def run_streamlit_app():
    if st is None:
        print("Streamlit is not installed. Run the terminal demo instead.")
        return

    st.set_page_config(page_title="Crop Monitoring Dashboard", layout="wide")
    st.title("Crop Monitoring Dashboard")
    st.write("AI-based crop status prediction using satellite-inspired features")

    data, encoder = build_sample_dataset()
    model = train_crop_model(data, encoder)

    with st.form("crop_form"):
        ndvi = st.slider("NDVI", 0.0, 1.0, 0.7)
        soil_moisture = st.slider("Soil Moisture", 0.0, 1.0, 0.6)
        temperature = st.slider("Temperature (°C)", 15, 40, 28)
        precipitation = st.slider("Precipitation (mm)", 0, 150, 40)
        growth_stage = st.selectbox("Growth Stage", ["vegetative", "flowering", "maturity"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        features = {
            "ndvi": ndvi,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "precipitation": precipitation,
            "growth_stage": growth_stage,
        }
        prediction = predict_crop_status(model, encoder, features)
        st.success(f"Predicted crop status: {prediction}")


if __name__ == "__main__":
    run_demo()
