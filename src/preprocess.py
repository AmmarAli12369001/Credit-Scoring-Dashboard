import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

COLUMN_NAMES = [
    'checking_account', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings', 'employment', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installments', 'housing', 'existing_credits',
    'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
]

SELECTED_FEATURES = [
    'checking_account', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings', 'employment', 'installment_rate',
    'age', 'housing', 'existing_credits', 'job', 'property'
]

CATEGORICAL_COLS = [
    'checking_account', 'credit_history', 'purpose',
    'savings', 'employment', 'housing', 'job', 'property'
]

NUMERICAL_COLS = ['duration', 'credit_amount', 'installment_rate', 'age', 'existing_credits']


def load_and_preprocess(filepath, save_scaler=True):
    df = pd.read_csv(filepath, sep=' ', header=None, names=COLUMN_NAMES)
    df['target'] = df['target'].map({1: 1, 2: 0})

    X = df[SELECTED_FEATURES].copy()
    y = df['target'].copy()

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{y.value_counts()}")

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X[NUMERICAL_COLS] = scaler.fit_transform(X[NUMERICAL_COLS])

    import os
    os.makedirs('models', exist_ok=True)
    if save_scaler:
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump((scaler, NUMERICAL_COLS, encoders), f)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    print(f"After SMOTE — Class distribution:\n{pd.Series(y_res).value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, SELECTED_FEATURES


def preprocess_input(input_dict, scaler_path='models/scaler.pkl'):
    with open(scaler_path, 'rb') as f:
        scaler, numerical_cols, encoders = pickle.load(f)

    input_df = pd.DataFrame([input_dict])[SELECTED_FEATURES]

    for col in CATEGORICAL_COLS:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    return input_df