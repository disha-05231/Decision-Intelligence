
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# ======================================
# LOAD DATA
# ======================================

features_df = pd.read_csv(
    "data/processed/features.csv"
)

clustered_df = pd.read_csv(
    "data/processed/clustered_data.csv"
)

bias_df = pd.read_csv(
    "data/processed/bias_analysis.csv"
)

action_summary = pd.read_csv(
    "data/processed/action_summary.csv"
)



# ======================================
# KPI SECTION
# ======================================

total_users = clustered_df['userid'].nunique()

total_sessions = len(clustered_df)

avg_clicks = (
    clustered_df['click_frequency']
    .mean()
)

abandonment_rate = (
    clustered_df['cart_abandonment']
    .mean() * 100
)

st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Users",
        total_users
    )

with col2:
    st.metric(
        "Total Sessions",
        total_sessions
    )

with col3:
    st.metric(
        "Average Clicks",
        round(avg_clicks, 2)
    )

with col4:
    st.metric(
        "Cart Abandonment %",
        f"{abandonment_rate:.2f}%"
    )

st.divider()

# ======================================
# CHART ROW 1
# ======================================

col1, col2 = st.columns(2)

# ======================================
# USER SEGMENTS
# ======================================

with col1:

    st.subheader("User Segments")

    segment_counts = (
        clustered_df['user_segment']
        .value_counts()
    )

    fig1 = px.bar(
        x=segment_counts.index,
        y=segment_counts.values,

        labels={
            'x': 'User Segment',
            'y': 'Count'
        },

        title="User Segment Distribution",

        color=segment_counts.values,

        color_continuous_scale='blues'
    )

    fig1.update_layout(
        template='plotly_dark',

        height=420,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        title_font=dict(
        size=20,
        color="white"
    ),

        font=dict(
        color="white"
    ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )
    

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.info(
    """
    Segments were generated using K-Means clustering on behavioral features:
    click frequency, engagement score, product switching behavior,
    interaction density and purchase intent score.
    """
)


# ======================================
# BEHAVIORAL FEATURE DISTRIBUTION
# ======================================

with col2:

    st.subheader("Behavioral Feature Distribution")

    feature_distribution = pd.DataFrame({

        "Feature": [

            "Click Frequency",

            "Product Switching",

            "Engagement Score",

            "Interaction Density"

        ],

        "Average Value": [

            clustered_df["click_frequency"].mean(),

            clustered_df["product_switch_count"].mean(),

            clustered_df["engagement_score"].mean(),

            clustered_df["interaction_density"].mean()

        ]
    })
    feature_distribution["Scaled Value"] = (
    MinMaxScaler()
    .fit_transform(
        feature_distribution[
            ["Average Value"]
        ]
    )
)

    fig2 = px.bar(

        feature_distribution,

        x="Feature",

        y="Scaled Value",

        title="Behavioral Feature Distribution",

        color="Average Value",

        color_continuous_scale="viridis"

    )

    fig2.update_layout(

        template="plotly_dark",

        height=420,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        title_font=dict(
        size=20,
        color="white"
    ),

        font=dict(
        color="white"
    ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )

    )
    

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info(
"""
The chart shows normalized average values of behavioral features used by the purchase prediction model. Values are scaled between 0 and 1 for comparison purposes.
"""
)

st.divider()


# ======================================
# CHART ROW 2
# ======================================

col3, col4 = st.columns(2)

# ======================================
# BIAS ANALYSIS
# ======================================

with col3:

    st.subheader("Behavioral Pattern Detection")

    bias_counts = bias_df[
    [
        "loss_aversion",
        "comparison_overload",
        "decision_fatigue",
        "impulsive_behavior"
    ]
].sum()
   
    bias_df = pd.DataFrame({

    "Pattern":[
        "Loss Aversion",
        "Comparison Overload",
        "Decision Fatigue",
        "Impulsive Behavior"
    ],

    "Count": bias_counts.values

})
    


   
    fig3 = px.bar(

    bias_df,

    x="Pattern",

    y="Count",

    title="Behavioral Pattern Detection",

    color="Count",

    color_continuous_scale="Reds"
)

    fig3.update_layout(

        template='plotly_dark',

        height=420,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        title_font=dict(
        size=20,
        color="white"
    ),

        font=dict(
        color="white"
    ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    fig3.update_traces(
    texttemplate='%{y}',
    textposition='outside'
)
    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.info("""
Behavioral patterns are identified using rule-based analysis of user interactions, session behavior and purchasing actions.

• Loss Aversion:
Cart abandonment after high engagement.

• Comparison Overload:
Frequent product switching.

• Decision Fatigue:
Long sessions with low conversions.

• Impulsive Behavior:
Fast purchase actions with low hesitation.
""")
    

# ======================================
# CART ABANDONMENT
# ======================================

with col4:

    st.subheader("Cart Abandonment")

    abandonment_counts = (
        clustered_df['cart_abandonment']
        .value_counts()
        .sort_index()
    )

    fig4 = px.pie(

        names=[
            "Completed",
            "Abandoned"
        ],

        values=[
            abandonment_counts[0],
            abandonment_counts[1]
        ],

        title="Cart Abandonment"
    )

    # Make labels clearly visible
    fig4.update_traces(

        textposition="inside",

        textinfo="label+percent",

        textfont=dict(
            size=16,
            color="white"
        )
    )

    fig4.update_layout(

        template='plotly_dark',

        height=420,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        font=dict(
            color="white",
            size=14
        ),

        title_font=dict(
            size=20,
            color="white"
        ),

        legend=dict(
            font=dict(
                size=14,
                color="white"
            )
        ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.info(
    f"""
    Cart abandonment represents sessions where users added products
    but exited without completing a purchase.

    Current cart abandonment rate is {abandonment_rate:.1f}%.

    A lower abandonment rate generally indicates a smoother
    customer journey and higher conversion effectiveness.
    """
    )

st.divider()

avg_purchase_intent = (
    clustered_df[
        "purchase_intent_score"
    ].mean()
)

st.subheader("Key Business Insights")

loyal_user_share = (
    segment_counts["Loyal Users"]
    / segment_counts.sum()
) * 100

st.success(f"""
• Total Users Analysed: {total_users}

• Cart Abandonment Rate: {abandonment_rate:.1f}%

• Loyal User Share: {loyal_user_share:.1f}%

• Average Purchase Intent Index: {avg_purchase_intent:.4f}

• Dominant User Segment: {segment_counts.idxmax()}
({segment_counts.max()} users)
""")