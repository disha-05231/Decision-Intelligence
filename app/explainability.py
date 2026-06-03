import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ======================================
# PAGE TITLE
# ======================================

st.markdown(
    "<h1 style='margin-top:0;'>Explainable AI Insights</h1>",
    unsafe_allow_html=True
)

# ======================================
# SHAP IMAGE
# ======================================

st.subheader("SHAP Feature Importance")

col1, col2 = st.columns([3, 1])

with col1:

    st.image(
        "reports/shap_summary.png",
        width=900
    )

# ======================================
# CALCULATE FEATURE IMPORTANCE
# ======================================

try:

    model = joblib.load(
        "models/trained_models/purchase_model.pkl"
    )

    df = pd.read_csv(
        "data/processed/features.csv"
    )

    feature_columns = [

        'click_frequency',
        'product_switch_count',
        'engagement_score',
        'interaction_density',
        'purchase_intent_score',
        'hesitation_score',
        'clicks_per_minute',
        'switch_rate',
        'engagement_ratio',
        'abandonment_risk'

    ]

    X = df[feature_columns]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    })



    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df["Percentage"] = (
        importance_df["Importance"]
        /
        importance_df["Importance"].sum()
    ) * 100

except Exception as e:

    st.error(f"SHAP Error: {e}")

    importance_df = pd.DataFrame()

# ======================================
# TOP FEATURES
# ======================================

with col2:

    st.subheader("Top Features")

    if not importance_df.empty:

        for i in range(min(5, len(importance_df))):

            feature = (
                importance_df.iloc[i]["Feature"]
                .replace("_", " ")
                .title()
            )

            pct = (
                importance_df.iloc[i]["Percentage"]
            )

            st.metric(
            label=f"#{i+1} {feature}",
            value=f"{pct:.1f}%"
)

# ======================================
# MODEL INTERPRETATION
# ======================================

st.subheader("Model Interpretation")

if not importance_df.empty:

    top_feature = (
        importance_df.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    second_feature = (
        importance_df.iloc[1]["Feature"]
        .replace("_", " ")
        .title()
    )

    third_feature = (
        importance_df.iloc[2]["Feature"]
        .replace("_", " ")
        .title()
    )

    st.info(f"""
The SHAP analysis explains how the machine learning model makes purchase predictions.

Top feature influencing predictions:
**{top_feature}**

Other major contributors:
• {second_feature}
• {third_feature}

Positive SHAP values push predictions toward purchase completion,
while negative SHAP values push predictions toward non-purchase behavior.
The distance from zero indicates the strength of influence.
""")

# ======================================
# BUSINESS IMPACT
# ======================================

st.subheader("Business Impact")

if not importance_df.empty:

    st.success(f"""
The model primarily relies on **{top_feature}** when predicting purchase behavior.

This suggests that optimizing user journeys around this behavioral factor could have the greatest impact on conversion rates.

Explainable AI allows stakeholders to understand why predictions are generated instead of treating the model as a black box.
""")

