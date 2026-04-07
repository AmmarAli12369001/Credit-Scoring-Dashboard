import shap
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


FEATURE_LABELS = {
    'checking_account': 'Checking Account Status',
    'duration': 'Loan Duration',
    'credit_history': 'Credit History',
    'purpose': 'Loan Purpose',
    'credit_amount': 'Credit Amount',
    'savings': 'Savings Account',
    'employment': 'Employment Duration',
    'installment_rate': 'Installment Rate (%)',
    'age': 'Age',
    'housing': 'Housing Status',
    'existing_credits': 'Existing Credits',
    'job': 'Job Type',
    'property': 'Property Owned'
}


def load_model(model_path='models/xgboost_model.pkl'):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_shap_values(input_df):
    model = load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    expected_value = explainer.expected_value
    return shap_values, expected_value, explainer


def get_top_reasons(input_df, top_n=5):
    shap_values, _, _ = get_shap_values(input_df)
    values = shap_values[0]
    features = input_df.columns.tolist()

    paired = list(zip(features, values))
    sorted_pairs = sorted(paired, key=lambda x: abs(x[1]), reverse=True)[:top_n]

    reasons = []
    for feature, value in sorted_pairs:
        label = FEATURE_LABELS.get(feature, feature)
        if value > 0:
            direction = "positively ↑"
            color = "green"
        else:
            direction = "negatively ↓"
            color = "red"
        reasons.append({
            "feature": label,
            "value": round(float(value), 4),
            "direction": direction,
            "color": color,
            "text": f"**{label}** influenced your score {direction} (impact: {value:+.3f})"
        })

    return reasons


def generate_waterfall_chart(input_df, save_path='app/shap_waterfall.png'):
    model = load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    renamed = input_df.rename(columns=FEATURE_LABELS)
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0].values,
        feature_names=[FEATURE_LABELS.get(c, c) for c in input_df.columns]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    shap.plots.waterfall(shap_explanation, show=False, max_display=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    return save_path


def generate_bar_chart(input_df, save_path='app/shap_bar.png'):
    shap_values, _, _ = get_shap_values(input_df)
    features = [FEATURE_LABELS.get(c, c) for c in input_df.columns]
    values = shap_values[0]

    sorted_idx = np.argsort(np.abs(values))[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_values = [values[i] for i in sorted_idx]

    colors = ['#22c55e' if v > 0 else '#ef4444' for v in sorted_values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    bars = ax.barh(sorted_features[::-1], sorted_values[::-1], color=colors[::-1], height=0.6)

    ax.set_xlabel('SHAP Value (Impact on Prediction)', color='#94a3b8', fontsize=10)
    ax.set_title('Feature Impact on Your Credit Score', color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='#475569', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    return save_path
