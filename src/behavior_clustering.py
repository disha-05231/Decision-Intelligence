import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_features(path):
    """
    Load engineered features
    """

    df = pd.read_csv(path)

    print("✅ Features dataset loaded")

    return df


def scale_features(df):
    """
    Scale numerical features
    """

    feature_columns = [
        'session_duration',
        'click_frequency',
        'product_switch_count',
        'cart_abandonment'
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df[feature_columns]
    )

    # Save scaler
    os.makedirs(
        "models/trained_models",
        exist_ok=True
    )

    joblib.dump(
        scaler,
        "models/trained_models/scaler.pkl"
    )

    print("✅ Features scaled successfully")

    return scaled_data


def train_kmeans(scaled_data):
    """
    Train KMeans clustering model
    """

    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    clusters = kmeans.fit_predict(
        scaled_data
    )

    # Save model
    joblib.dump(
        kmeans,
        "models/trained_models/kmeans_model.pkl"
    )

    print("✅ KMeans model trained")

    return kmeans, clusters


def assign_segment_names(df):
    """
    Convert cluster numbers to business-friendly names
    """

    segment_mapping = {
        0: "Explorers",
        1: "Impulsive Buyers",
        2: "Loyal Users",
        3: "Hesitant Users"
    }

    df['user_segment'] = (
        df['cluster']
        .map(segment_mapping)
    )

    return df


def save_clustered_data(df, output_path):
    """
    Save clustered dataset
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print("✅ Clustered data saved")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🧠 USER BEHAVIOR CLUSTERING")
    print("=" * 60)

    input_path = (
        "data/processed/features.csv"
    )

    output_path = (
        "data/processed/clustered_data.csv"
    )

    # Load features
    df = load_features(input_path)

    # Scale features
    scaled_data = scale_features(df)

    # Train clustering model
    kmeans, clusters = train_kmeans(
        scaled_data
    )

    # Add cluster labels
    df['cluster'] = clusters

    # Assign business names
    df = assign_segment_names(df)

    # Save clustered dataset
    save_clustered_data(
        df,
        output_path
    )

    print(
        "\n🎉 User segmentation completed successfully!"
    )