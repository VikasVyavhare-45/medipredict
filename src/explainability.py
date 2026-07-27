"""
explainability.py
Reusable SHAP explanation for a single patient's prediction, for any of the
10 diseases. Meant to be called from the Streamlit dashboard after a
prediction is made, to show "why" the model predicted what it did.

Like predict.py, this reads column order from scaler.feature_names_in_
so it never drifts out of sync with what was used at training time.
"""

import joblib
import numpy as np
import pandas as pd
import shap

import os

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:
    def _cache_resource(func):
        return func

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

TREE_MODELS = ["random_forest", "xgboost", "decision_tree"]

# Same best-model choices as predict.py - keep these two files in sync.
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

# Cache so we don't rebuild a KernelExplainer's background sample on every call
_explainer_cache = {}


@_cache_resource
def _load_model_and_scaler(disease, model_name=None):
    model_name = model_name or BEST_MODEL[disease]
    model = joblib.load(f"{MODELS_DIR}/{disease}/{model_name}.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/{disease}/scaler.pkl")
    return model, scaler, model_name


def _get_explainer(disease, model, model_name, background_data=None):
    """
    Builds (or returns a cached) SHAP explainer for this disease+model.
    background_data: scaled numpy array used as the KernelExplainer background
    sample, only needed the first time for non-tree models. Pass a slice of
    your training data (e.g. X_train_scaled[:100]) if the model isn't tree-based.
    """
    cache_key = f"{disease}:{model_name}"
    if cache_key in _explainer_cache:
        return _explainer_cache[cache_key]

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
    else:
        if background_data is None:
            raise ValueError(
                f"'{model_name}' is not a tree model - pass background_data "
                f"(a sample of scaled training rows) the first time you call "
                f"explain_prediction() for '{disease}'."
            )
        background = shap.sample(background_data, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    _explainer_cache[cache_key] = explainer
    return explainer


def explain_prediction(disease, input_dict, model_name=None, background_data=None, top_n=None):
    """
    input_dict: {feature_name: value, ...} - same dict you'd pass to predict_disease()
    background_data: only required the first call for a non-tree best model
                      (svm/logistic_regression/knn/naive_bayes)
    top_n: if set, only return the top_n features by absolute impact

    Returns a DataFrame with columns: feature, shap_value, abs_impact
    - positive shap_value -> pushed the prediction toward "disease present"
    - negative shap_value -> pushed the prediction toward "no disease"
    """
    model, scaler, model_name = _load_model_and_scaler(disease, model_name)

    order = list(scaler.feature_names_in_)

    from predict import _apply_missing_value_fix
    input_dict = _apply_missing_value_fix(disease, input_dict)

    missing = [f for f in order if f not in input_dict]
    if missing:
        raise ValueError(f"Missing input fields for {disease}: {missing}")

    row_df = pd.DataFrame([[input_dict[f] for f in order]], columns=order)
    row_scaled = scaler.transform(row_df)

    explainer = _get_explainer(disease, model, model_name, background_data)

    if model_name in TREE_MODELS:
        sv = explainer.shap_values(row_scaled)
        sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # class 1 = disease present (newer SHAP versions)
        elif isinstance(sv, list):
            sv = sv[1]
        sv = sv.reshape(-1)
    else:
        sv = explainer.shap_values(row_scaled)
        sv = np.array(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        elif isinstance(sv, list):
            sv = sv[1]
        sv = sv.reshape(-1)

    impact_df = pd.DataFrame({"feature": order, "shap_value": sv})
    impact_df["abs_impact"] = impact_df["shap_value"].abs()
    impact_df = impact_df.sort_values("abs_impact", ascending=False)

    if top_n:
        impact_df = impact_df.head(top_n)

    return impact_df.reset_index(drop=True)


if __name__ == "__main__":
    # quick manual test (diabetes uses random_forest, a tree model - no background_data needed)
    sample_input = {
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 78,
        "SkinThickness": 25, "Insulin": 100, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45, "Age": 40
    }
    explanation = explain_prediction("diabetes", sample_input, top_n=5)
    print(explanation)
