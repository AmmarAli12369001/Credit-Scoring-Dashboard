import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import predict
from explain import get_top_reasons, generate_waterfall_chart, generate_bar_chart

st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background-color: #0f172a; }
.block-container { padding: 2rem 2rem 2rem 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e293b;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Header */
.dashboard-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border: 1px solid #1d4ed8;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.dashboard-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.dashboard-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

/* Score card */
.score-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.score-number {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.score-label { font-size: 1.5rem; margin-top: 0.5rem; font-weight: 600; }
.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.8rem;
    text-transform: uppercase;
}
.risk-low    { background: #14532d; color: #4ade80; border: 1px solid #16a34a; }
.risk-medium { background: #713f12; color: #fbbf24; border: 1px solid #d97706; }
.risk-high   { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

/* Reason card */
.reason-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.reason-positive { border-left: 3px solid #22c55e; }
.reason-negative { border-left: 3px solid #ef4444; }
.reason-feature { color: #f1f5f9; font-weight: 600; font-size: 0.92rem; }
.reason-impact  { color: #64748b; font-size: 0.82rem; margin-top: 0.2rem; }
.impact-positive { color: #4ade80; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.impact-negative { color: #f87171; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

/* Section headers */
.section-title {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* Probability bar */
.prob-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 10px;
    margin-top: 0.5rem;
    overflow: hidden;
    border: 1px solid #334155;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

.info-box {
    background: #0f2744;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #93c5fd;
    font-size: 0.85rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📋 Applicant Details")
    st.markdown("---")

    age = st.slider("Age", min_value=18, max_value=75, value=35)

    credit_amount = st.number_input(
        "Credit Amount (€)", min_value=500, max_value=20000, value=5000, step=500
    )

    duration = st.slider("Loan Duration (Months)", min_value=6, max_value=72, value=24)

    checking_account = st.selectbox(
        "Checking Account Status",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "< 0 € (Overdrawn)",
            2: "0–200 €",
            3: "> 200 € / Salary",
            4: "No Checking Account"
        }[x],
        index=1
    )

    savings = st.selectbox(
        "Savings Account",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "< 100 €",
            2: "100–500 €",
            3: "500–1000 €",
            4: "> 1000 €",
            5: "Unknown / No savings"
        }[x],
        index=1
    )

    employment = st.selectbox(
        "Employment Duration",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Unemployed",
            2: "< 1 Year",
            3: "1–4 Years",
            4: "4–7 Years",
            5: "> 7 Years"
        }[x],
        index=2
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "No Credits",
            1: "All Paid",
            2: "Existing Paid Duly",
            3: "Delay in Past",
            4: "Critical / Other Credits"
        }[x],
        index=2
    )

    purpose = st.selectbox(
        "Loan Purpose",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        format_func=lambda x: {
            0: "Car (New)",
            1: "Car (Used)",
            2: "Furniture / Equipment",
            3: "Radio / Television",
            4: "Domestic Appliances",
            5: "Repairs",
            6: "Education",
            7: "Vacation",
            8: "Retraining",
            9: "Business",
            10: "Others"
        }[x],
        index=3
    )

    housing = st.selectbox(
        "Housing Status",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Rent", 2: "Own", 3: "For Free"}[x],
        index=1
    )

    installment_rate = st.slider(
        "Installment Rate (% of income)", min_value=1, max_value=4, value=2
    )

    existing_credits = st.slider(
        "Number of Existing Credits", min_value=0, max_value=4, value=1
    )

    property_type = st.selectbox(
        "Property Owned",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Real Estate",
            2: "Savings / Insurance",
            3: "Car / Other",
            4: "Unknown / No Property"
        }[x],
        index=0
    )

    job = st.selectbox(
        "Job Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Unskilled (Non-Resident)",
            2: "Unskilled (Resident)",
            3: "Skilled / Official",
            4: "Management / Self-Employed"
        }[x],
        index=2
    )

    st.markdown("---")
    analyse_btn = st.button("🔍 Analyse Application")


st.markdown("""
<div class="dashboard-header">
    <p class="dashboard-title">📊 Credit Scoring Dashboard</p>
    <p class="dashboard-subtitle">Explainable AI Powered Credit Risk Assessment using SHAP</p>
</div>
""", unsafe_allow_html=True)


if not analyse_btn:
    st.markdown("""
    <div class="info-box">
        ℹ️ Fill in the applicant details in the sidebar and click <strong>Analyse Application</strong> to get a credit decision with full explanation.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="score-card">
            <div style="font-size:2rem">🤖</div>
            <div style="color:#f1f5f9; font-weight:600; margin-top:0.5rem">XGBoost Model</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:0.3rem">Trained on German Credit Dataset with 1000 samples</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="score-card">
            <div style="font-size:2rem">💡</div>
            <div style="color:#f1f5f9; font-weight:600; margin-top:0.5rem">SHAP Explanations</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:0.3rem">Every decision explained with feature-level impact scores</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="score-card">
            <div style="font-size:2rem">⚖️</div>
            <div style="color:#f1f5f9; font-weight:600; margin-top:0.5rem">Bias Aware</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:0.3rem">Transparent scoring to surface and reduce algorithmic bias</div>
        </div>""", unsafe_allow_html=True)

else:
    input_dict = {
        'checking_account': checking_account,
        'duration': duration,
        'credit_history': credit_history,
        'purpose': purpose,
        'credit_amount': credit_amount,
        'savings': savings,
        'employment': employment,
        'installment_rate': installment_rate,
        'age': age,
        'housing': housing,
        'existing_credits': existing_credits,
        'job': job,
        'property': property_type
    }

    with st.spinner("Analysing application..."):
        try:
            result = predict(input_dict)
        except FileNotFoundError:
            st.error("⚠️ Model not found. Please run `python src/train.py` first to train the model.")
            st.stop()

    label        = result["label"]
    credit_score = result["credit_score"]
    risk         = result["risk"]
    probability  = result["probability"]
    input_df     = result["input_df"]

    if risk == "Low":
        score_color = "#4ade80"
        risk_class  = "risk-low"
    elif risk == "Medium":
        score_color = "#fbbf24"
        risk_class  = "risk-medium"
    else:
        score_color = "#f87171"
        risk_class  = "risk-high"

    col_score, col_reasons = st.columns([1, 2])

    with col_score:
        st.markdown(f"""
        <div class="score-card">
            <div class="section-title">Credit Decision</div>
            <div class="score-number" style="color:{score_color}">{credit_score}</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.2rem">/ 100</div>
            <div class="score-label">{label}</div>
            <span class="risk-badge {risk_class}">{risk} Risk</span>
            <div style="margin-top:1.5rem;">
                <div style="color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;">Approval Probability</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{credit_score}%; background:{'linear-gradient(90deg,#16a34a,#4ade80)' if risk=='Low' else 'linear-gradient(90deg,#d97706,#fbbf24)' if risk=='Medium' else 'linear-gradient(90deg,#dc2626,#f87171)'};"></div>
                </div>
                <div style="color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:0.85rem; margin-top:0.4rem;">{probability:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_reasons:
        st.markdown('<div class="section-title">Top Factors Affecting Your Score</div>', unsafe_allow_html=True)
        reasons = get_top_reasons(input_df, top_n=6)
        for r in reasons:
            border_class = "reason-positive" if r["value"] > 0 else "reason-negative"
            impact_class = "impact-positive" if r["value"] > 0 else "impact-negative"
            impact_sign  = f"+{r['value']:.3f}" if r["value"] > 0 else f"{r['value']:.3f}"
            icon = "↑" if r["value"] > 0 else "↓"
            st.markdown(f"""
            <div class="reason-card {border_class}">
                <div style="flex:1">
                    <div class="reason-feature">{r['feature']}</div>
                    <div class="reason-impact">{r['direction']}</div>
                </div>
                <div class="{impact_class}">{icon} {impact_sign}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_bar, col_waterfall = st.columns(2)

    with col_bar:
        st.markdown('<div class="section-title">Feature Impact (Bar Chart)</div>', unsafe_allow_html=True)
        with st.spinner("Generating chart..."):
            bar_path = generate_bar_chart(input_df, save_path='app/shap_bar.png')
        st.image(bar_path, use_column_width=True)

    with col_waterfall:
        st.markdown('<div class="section-title">SHAP Waterfall Explanation</div>', unsafe_allow_html=True)
        with st.spinner("Generating waterfall..."):
            wf_path = generate_waterfall_chart(input_df, save_path='app/shap_waterfall.png')
        st.image(wf_path, use_column_width=True)

    with st.expander("📋 View Submitted Application Details"):
        import pandas as pd
        display_dict = {
            "Age": age,
            "Credit Amount (€)": credit_amount,
            "Duration (months)": duration,
            "Checking Account": checking_account,
            "Savings Account": savings,
            "Employment": employment,
            "Credit History": credit_history,
            "Purpose": purpose,
            "Housing": housing,
            "Installment Rate (%)": installment_rate,
            "Existing Credits": existing_credits,
            "Property": property_type,
            "Job Type": job
        }
        df_display = pd.DataFrame(list(display_dict.items()), columns=["Feature", "Value"])
        st.dataframe(df_display, use_container_width=True, hide_index=True)