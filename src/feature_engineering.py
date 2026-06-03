import pandas as pd
import os


def load_processed_data(path):
    """
    Load cleaned dataset
    """

    df = pd.read_csv(path)

    print("✅ Processed dataset loaded successfully")

    return df


def calculate_session_duration(df):
    """
    Calculate realistic session duration
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort data properly
    df = df.sort_values(
        by=['userid', 'sessionid', 'timestamp']
    )

    session_duration = (
        df.groupby(['userid', 'sessionid'])['timestamp']
        .agg(['min', 'max'])
        .reset_index()
    )

    session_duration['session_duration'] = (
        session_duration['max'] - session_duration['min']
    ).dt.total_seconds()

    # Cap unrealistic durations
    session_duration['session_duration'] = (
        session_duration['session_duration']
        .clip(upper=3600)
    )
    print(
    session_duration['session_duration']
    .describe()
)

    return session_duration[
        ['userid', 'sessionid', 'session_duration']
    ]


def calculate_click_frequency(df):
    """
    Count events per session
    """

    click_frequency = (
        df.groupby(['userid', 'sessionid'])
        .size()
        .reset_index(name='click_frequency')
    )

    return click_frequency


def calculate_product_switching(df):
    """
    Count unique products viewed in session
    """

    switching = (
        df.groupby(['userid', 'sessionid'])['productid']
        .nunique()
        .reset_index(name='product_switch_count')
    )

    return switching


def calculate_purchase_ratio(df):
    """
    Calculate purchase ratio per session
    """

    purchase_events = df[
        df['eventtype'] == 'purchase'
    ]

    purchase_ratio = (
        purchase_events.groupby(
            ['userid', 'sessionid']
        )
        .size()
        .reset_index(name='purchase_count')
    )

    total_events = (
        df.groupby(['userid', 'sessionid'])
        .size()
        .reset_index(name='total_events')
    )

    merged = pd.merge(
        total_events,
        purchase_ratio,
        on=['userid', 'sessionid'],
        how='left'
    )

    merged['purchase_count'] = (
        merged['purchase_count']
        .fillna(0)
    )

    merged['purchase_ratio'] = (
        merged['purchase_count'] /
        merged['total_events']
    )

    return merged[
        ['userid', 'sessionid', 'purchase_ratio']
    ]


def calculate_cart_abandonment(df):
    """
    Detect cart abandonment
    """

    cart_sessions = df[
        df['eventtype'] == 'add_to_cart'
    ][['userid', 'sessionid']].drop_duplicates()

    purchase_sessions = df[
        df['eventtype'] == 'purchase'
    ][['userid', 'sessionid']].drop_duplicates()

    purchase_set = set(
        zip(
            purchase_sessions['userid'],
            purchase_sessions['sessionid']
        )
    )

    abandonment_data = []

    for _, row in cart_sessions.iterrows():

        key = (row['userid'], row['sessionid'])

        abandoned = 1

        if key in purchase_set:
            abandoned = 0

        abandonment_data.append({
            'userid': row['userid'],
            'sessionid': row['sessionid'],
            'cart_abandonment': abandoned
        })

    abandonment_df = pd.DataFrame(
        abandonment_data
    )

    return abandonment_df


def merge_all_features(df):

    print("\n⚙️ Generating behavioral features...")

    session_duration = calculate_session_duration(df)

    click_frequency = calculate_click_frequency(df)

    product_switching = calculate_product_switching(df)

    purchase_ratio = calculate_purchase_ratio(df)

    cart_abandonment = calculate_cart_abandonment(df)

    # =========================
    # MERGE FEATURES
    # =========================

    features = pd.merge(
        session_duration,
        click_frequency,
        on=['userid', 'sessionid']
    )

    features = pd.merge(
        features,
        product_switching,
        on=['userid', 'sessionid']
    )

    features = pd.merge(
        features,
        purchase_ratio,
        on=['userid', 'sessionid']
    )

    features = pd.merge(
        features,
        cart_abandonment,
        on=['userid', 'sessionid'],
        how='left'
    )

    # =========================
    # FILL MISSING VALUES
    # =========================

    features['cart_abandonment'] = (
        features['cart_abandonment']
        .fillna(0)
    )

    # =========================
    # NEW BEHAVIORAL FEATURES
    # =========================

    # User engagement intensity
    features['engagement_score'] = (

        features['click_frequency']
        / (
            features['session_duration'] + 1
        )

    )

    # Product exploration behavior
    features['interaction_density'] = (

        features['product_switch_count']
        / (
            features['click_frequency'] + 1
        )

    )

    # Purchase tendency
    features['purchase_intent_score'] = (

    features['click_frequency']

    /

    (
        features['product_switch_count'] + 1
    )

)

    # Hesitation behavior
    features['hesitation_score'] = (

    features['session_duration'] / 60

    )/\
    (
    features['click_frequency'] + 1
    )

# ======================================
# ADVANCED BEHAVIORAL FEATURES
# ======================================

# Click speed
    features['clicks_per_minute'] = (

        features['click_frequency']
    /
    (
        (features['session_duration'] / 60) + 1
    )

)

# Product switching rate
    features['switch_rate'] = (

        features['product_switch_count']
    /
    (
        features['click_frequency'] + 1
    )

)

# Engagement efficiency
    features['engagement_ratio'] = (

        features['engagement_score']
    /
    (
        features['interaction_density'] + 0.001
    )

)

    # ======================================
# NEW ADVANCED FEATURES
# ======================================

    features['abandonment_risk'] = (

    features['interaction_density']
    *
    features['hesitation_score']

)

    features['engagement_quality'] = (

        features['engagement_score']
        *
        features['purchase_ratio']

)

    features['interaction_efficiency'] = (

        features['purchase_ratio']
        /
    (
        features['interaction_density'] + 0.00001
    )

)

    print("✅ Feature engineering completed")

    return features


def save_features(df, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print("\n✅ Features saved successfully")
    print(f"📁 {output_path}")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🧠 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    input_path = (
        "data/processed/processed_data.csv"
    )

    output_path = (
        "data/processed/features.csv"
    )

    # Load processed dataset
    df = load_processed_data(input_path)

    # Generate behavioral features
    features = merge_all_features(df)

    # Save features
    save_features(features, output_path)

    print(
        "\n🎉 Feature engineering completed successfully!"
    )