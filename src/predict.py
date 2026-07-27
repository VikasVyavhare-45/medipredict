"""
predict.py
Loads a saved (model, scaler) pair for a given disease and returns a
prediction + confidence score for a single patient's input.

Feature order is NOT hardcoded here. Because each notebook fits the
scaler on a pandas DataFrame (`scaler.fit_transform(X_train)` where
X_train is a DataFrame with column names), scikit-learn automatically
stores that order on the scaler itself as `scaler.feature_names_in_`.
We read it from there, so it can never drift out of sync with training.
"""

import joblib
import numpy as np
import pandas as pd

import os

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:
    # Allow this module to still be imported/run outside Streamlit
    # (e.g. the __main__ test block below, or a plain notebook cell).
    def _cache_resource(func):
        return func

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Best model chosen per disease based on your results_df F1 comparison.
BEST_MODEL = {
    "diabetes": "random_forest",
    "heart": "random_forest",
    "parkinsons": "xgboost",
    "liver": "random_forest",
    "kidney": "random_forest",
    "breast_cancer": "svm",
    "stroke": "random_forest",
    "hepatitis": "random_forest",
    "thyroid": "random_forest",
    "lung_cancer": "random_forest",
}

# During training, these diabetes columns had 0 treated as "missing" and
# replaced with the column median (0 isn't physiologically valid for
# Glucose/BloodPressure/etc). If a live prediction gets a literal 0 for
# these fields (e.g. "I don't know my insulin level"), the scaler - fit
# on the median-imputed training data - will scale that 0 to an extreme
# out-of-distribution value and can flip the prediction. We apply the same
# substitution here so live inputs match what the model actually learned
# from. These are the standard reported medians for the public Pima
# Indians Diabetes dataset - if your notebook computed slightly different
# values (df[col].median() after replacing 0 -> NaN, before training),
# swap them in here for exact parity.
ZERO_AS_MISSING = {
    "diabetes": {
        "Glucose": 117.0,
        "BloodPressure": 72.0,
        "SkinThickness": 23.0,
        "Insulin": 125.0,
        "BMI": 32.3,
    },
}


def _apply_missing_value_fix(disease, input_dict):
    fixes = ZERO_AS_MISSING.get(disease)
    if not fixes:
        return input_dict
    cleaned = dict(input_dict)
    for col, median_val in fixes.items():
        if col in cleaned and cleaned[col] == 0:
            cleaned[col] = median_val
    return cleaned


@_cache_resource
def load_model_and_scaler(disease, model_name=None):
    """
    Cached with st.cache_resource: joblib.load() reads from disk every
    call, and every widget click on a disease page triggers a full script
    rerun. Without caching, that meant re-reading (and for some models,
    re-deserializing tens of MB of) .pkl files from disk on every single
    click - the main cause of pages feeling slow to respond. Streamlit
    keeps the returned (model, scaler) in memory across reruns, keyed by
    the (disease, model_name) arguments, so this now only hits disk once
    per model for the whole session.
    """
    model_name = model_name or BEST_MODEL[disease]
    model = joblib.load(f"{MODELS_DIR}/{disease}/{model_name}.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/{disease}/scaler.pkl")
    return model, scaler


def predict_disease(disease, input_dict, model_name=None):
    """
    input_dict: {feature_name: value, ...} - keys must match the columns
    the scaler was trained on (order doesn't matter, we reorder for you).
    Returns: {"prediction": 0/1, "confidence": float (0-100), "model_used": str}
    """
    model, scaler = load_model_and_scaler(disease, model_name)

    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError(
            f"scaler for '{disease}' has no feature_names_in_ - it was likely "
            f"fit on a numpy array instead of a DataFrame. Re-fit it on X_train "
            f"(a DataFrame) in the training notebook and re-save the scaler."
        )

    order = list(scaler.feature_names_in_)

    input_dict = _apply_missing_value_fix(disease, input_dict)

    missing = [f for f in order if f not in input_dict]
    if missing:
        raise ValueError(f"Missing input fields for {disease}: {missing}")

    # Build as a DataFrame with the exact training column order/names,
    # so scaler.transform sees matching columns regardless of dict order.
    row_df = pd.DataFrame([[input_dict[f] for f in order]], columns=order)
    row_scaled = scaler.transform(row_df)

    prediction = int(model.predict(row_scaled)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        confidence = float(max(proba) * 100)
    else:
        confidence = 100.0  # fallback if model has no predict_proba

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "model_used": model_name or BEST_MODEL[disease],
    }


if __name__ == "__main__":
    # quick manual test
    sample_input = {
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 78,
        "SkinThickness": 25, "Insulin": 100, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 40
    }
    result = predict_disease("diabetes", sample_input)
    print(result)
