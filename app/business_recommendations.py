import streamlit as st
import pandas as pd

# ======================================
# LOAD DATA
# ======================================

clustered_df = pd.read_csv(
    "data/processed/clustered_data.csv"
)

# ======================================
# METRICS
# ======================================

total_users = len(clustered_df)

abandonment_rate = (
    clustered_df["cart_abandonment"].mean() * 100
)

segment_counts = (
    clustered_df["user_segment"]
    .value_counts()
)

dominant_segment = (
    segment_counts.idxmax()
)

dominant_segment_count = (
    segment_counts.max()
)

loyal_user_share = (
    segment_counts.get("Loyal Users", 0)
    / total_users
) * 100

avg_purchase_intent = (
    clustered_df["purchase_intent_score"]
    .mean()
)

avg_hesitation = (
    clustered_df["hesitation_score"]
    .mean()
)

# ======================================
# PAGE TITLE
# ======================================

st.markdown(
    "<h1 style='margin-top:0;'>Strategic Business Insights</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='font-size:18px;'>
    AI-generated executive summary derived from behavioral analytics,
    user segmentation and purchase prediction insights.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ======================================
# KEY RISK
# ======================================

st.subheader("Key Business Risk")

if abandonment_rate > 20:

    risk_text = f"""
    Cart abandonment rate is currently
    {abandonment_rate:.1f}%.

    A significant number of users leave
    the purchasing journey before completing
    checkout, potentially reducing revenue.
    """

elif abandonment_rate > 10:

    risk_text = f"""
    Cart abandonment rate is
    {abandonment_rate:.1f}%.

    Moderate checkout friction exists
    and should be monitored.
    """

else:

    risk_text = """
    Cart abandonment remains low and
    does not currently represent a major
    business concern.
    """

st.error(risk_text)

# ======================================
# KEY OPPORTUNITY
# ======================================

st.subheader("Key Growth Opportunity")

if loyal_user_share > 20:

    opportunity_text = f"""
    Loyal Users account for
    {loyal_user_share:.1f}% of the user base.

    This audience represents a strong
    opportunity for retention campaigns,
    loyalty rewards and upselling strategies.
    """

else:

    opportunity_text = """
    Loyal user concentration is currently low.

    Increasing customer retention should be
    a strategic priority.
    """

st.success(opportunity_text)

# ======================================
# DOMINANT USER SEGMENT
# ======================================

st.subheader("Dominant User Segment")

st.info(
    f"""
    The largest behavioral segment is
    '{dominant_segment}' with
    {dominant_segment_count} users.

    This segment currently represents the
    most influential customer behavior pattern
    within the platform.
    """
)

# ======================================
# USER BEHAVIOR ANALYSIS
# ======================================

st.subheader("Behavioral Analysis")

if dominant_segment == "Hesitant Users":

    behavior_text = """
    Users spend considerable time evaluating
    products before making purchase decisions.

    This suggests decision friction,
    uncertainty and increased comparison behavior.
    """

elif dominant_segment == "Impulsive Buyers":

    behavior_text = """
    Users tend to make quick purchasing decisions.

    Marketing campaigns and urgency-based offers
    are likely to perform well for this segment.
    """

elif dominant_segment == "Loyal Users":

    behavior_text = """
    Existing customers demonstrate recurring
    engagement and purchasing behavior.

    Retention-focused strategies should be
    prioritized.
    """

else:

    behavior_text = """
    Users frequently explore products and
    categories before committing to purchases.

    Discovery and recommendation systems may
    improve conversions.
    """

st.warning(behavior_text)

# ======================================
# MODEL INSIGHT
# ======================================

st.subheader("AI Model Insight")

st.markdown(
    f"""
    • Total Users Analysed: **{total_users}**

    • Average Purchase Intent Index:
    **{avg_purchase_intent:.4f}**

    • Average Hesitation Score:
    **{avg_hesitation:.2f}**

    • Dominant Segment:
    **{dominant_segment}**
    """
)

# ======================================
# EXECUTIVE SUMMARY
# ======================================

st.subheader("Executive Summary")

summary = f"""
The Human Decision Intelligence Engine identified
'{dominant_segment}' as the dominant behavioral
group across the analysed sessions.

Cart abandonment currently stands at
{abandonment_rate:.1f}%,
indicating that a noticeable portion of users
leave before completing purchases.

Behavioral analytics suggest that customer
decision-making patterns are heavily influenced
by hesitation, product evaluation and engagement
quality.

Based on the current behavioral landscape,
improving customer guidance, reducing decision
friction and strengthening retention strategies
represent the highest-impact business actions.
"""

st.success(summary)