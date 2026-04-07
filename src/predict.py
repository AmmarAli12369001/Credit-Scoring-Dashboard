import pickle
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_input


def load_model(model_path='models/xgboost_model.pkl'):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_feature_names(path='models/feature_names.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(input_dict):
    model = load_model()
    input_df = preprocess_input(input_dict)

    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    credit_score = round(probability * 100)

    if credit_score >= 65:
        risk = "Low"
        label = "Approved"
    elif credit_score >= 40:
        risk = "Medium"
        label = "Manual Review"
    else:
        risk = "High"
        label = "Rejected"

    return {
        "label": label,
        "credit_score": credit_score,
        "risk": risk,
        "probability": round(float(probability), 4),
        "input_df": input_df
    }
