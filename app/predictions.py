import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap

# ======================================
# LOAD MODEL
# ======================================

model = joblib.load(
    "models/trained_models/purchase_model.pkl"
)

# ======================================
# PAGE CONFIG
# ======================================

st.markdown(
    "<h1 style='margin-top:0;'>Purchase Behavior Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "Analyze user behavioral patterns and predict purchase intent."
)


# ======================================
# INPUT SECTION
# ======================================

st.subheader("Behavioral Inputs")

col1, col2 = st.columns(2)

with col1:

    click_frequency = st.slider(
        "Click Frequency",
        1,
        50,
        10
    )

    product_switch_count = st.slider(
        "Product Switching",
        1,
        20,
        5
    )

    engagement_score = st.slider(
        "Engagement Score",
        0.0,
        1.0,
        0.2
    )

with col2:

    interaction_density = st.slider(
        "Interaction Density",
        0.0,
        2.0,
        0.5
    )

    
    hesitation_score = st.slider(
        "Hesitation Score",
        0.0,
        3000.0,
        500.0
    )



# ======================================
# PREDICTION
# ======================================

if st.button("Predict User Behavior"):

    clicks_per_minute = click_frequency / 61

    switch_rate = (
    product_switch_count /
    (click_frequency + 1)
    )

    engagement_ratio = (
    engagement_score /
    (interaction_density + 0.001)
)

    abandonment_risk = (
    product_switch_count *
    hesitation_score
)
    purchase_intent_score = (
    click_frequency *
    interaction_density *
    engagement_score
)
    
    input_data = pd.DataFrame({

    "click_frequency": [click_frequency],

    "product_switch_count": [product_switch_count],

    "engagement_score": [engagement_score],

    "interaction_density": [interaction_density],

    "purchase_intent_score": [purchase_intent_score],

    "hesitation_score": [hesitation_score],

    "clicks_per_minute": [clicks_per_minute],

    "switch_rate": [switch_rate],

    "engagement_ratio": [engagement_ratio],

    "abandonment_risk": [abandonment_risk]

})
    

    prediction = model.predict(input_data)[0]

    probability = (
        model.predict_proba(input_data)[0][1]
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Impact": shap_values.values[0]
})

    shap_df["AbsImpact"] = shap_df["Impact"].abs()

    shap_df = shap_df.sort_values(
    by="AbsImpact",
    ascending=False
)

    st.subheader("Prediction Results")

    

    

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix":"%"},
            gauge={
                "axis":{
                    "range":[0,100]
            }
        }
    )
)

    fig.update_layout(
        height=300,
        paper_bgcolor="#111827"
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.markdown("---")

    st.info("""
Prediction Score represents the estimated likelihood that a user
will complete a purchase based on behavioral features.

The score is generated using the trained machine learning model
and ranges from 0% to 100%.
""")
    
    st.subheader("Top Factors")

    top_features = shap_df.reindex(
        shap_df["Impact"].abs().sort_values(
            ascending=False
    ).index
).head(3)

    col1, col2, col3 = st.columns(3)

    with col1:
        
        st.write(
        top_features.iloc[0]["Feature"]
        .replace("_", " ")
        .title()
    )

    with col2:
        
        st.write(
        top_features.iloc[1]["Feature"]
        .replace("_", " ")
        .title()
    )

    with col3:
        
        st.write(
        top_features.iloc[2]["Feature"]
        .replace("_", " ")
        .title()
    )
    st.markdown("---")

    
    
    # ======================================
    # BEHAVIORAL INSIGHTS
    # ======================================

    st.subheader("Behavioral Insights")

    insights = []

    if hesitation_score > 1200:
        insights.append("High hesitation before purchasing")

    if product_switch_count > 5:
        insights.append("Comparing multiple products")

    if engagement_score > 0.30:
        insights.append("Strong engagement behavior")

    if interaction_density < 0.30:
        insights.append("Low interaction activity")

# Fallback insight
    if len(insights) == 0:
        insights.append("No unusual behavioral patterns detected")
# Show only top 2 insights
    insights = insights[:2]

    col1, col2 = st.columns(2)

    if len(insights) >= 1:
        with col1:
            st.info(insights[0])

    if len(insights) >= 2:
        with col2:
            st.info(insights[1])

    st.subheader("Decision Summary")

    if probability >= 0.80:
        st.success(f"High Purchase Intent - {probability*100:.1f}%")

    elif probability >= 0.60:
        st.warning(f"Moderate Purchase Intent - {probability*100:.1f}%")

    else:
        st.error(f"Low Purchase Intent - {probability*100:.1f}%")

    st.markdown("---")

    st.subheader("Recommended Actions")

    actions = []

    if probability > 0.80:

        actions = [
        "Show premium products",
        "Offer loyalty rewards"
    ]

    elif probability > 0.60:

        actions = [
        "Provide personalized recommendations",
        "Offer limited-time discounts"
    ]

    else:

        actions = [
        "Send reminder notifications",
        "Offer introductory discounts"
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.info(actions[0])

    with col2:
        st.info(actions[1])