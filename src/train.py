import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score
)
from preprocess import load_and_preprocess


def train_model(data_path='data/raw/german.data'):
    print("=" * 50)
    print("   Credit Scoring Model — Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(data_path)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("   Evaluation Results")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad Credit', 'Good Credit']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    print("\nModel saved to models/xgboost_model.pkl")
    print("Feature names saved to models/feature_names.pkl")
    return model


if __name__ == '__main__':
    train_model()
