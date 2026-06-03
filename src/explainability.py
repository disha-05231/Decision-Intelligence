import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os


def load_model(model_path):

    model = joblib.load(model_path)

    print("✅ Trained model loaded")

    return model


def load_dataset(data_path):

    df = pd.read_csv(data_path)
    

    print("✅ Dataset loaded")

    return df


def prepare_features(df):

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

    return X


def generate_shap_values(model, X):


    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    print("✅ SHAP values generated")

    return shap_values


def plot_shap_summary(shap_values, X):

    os.makedirs(
        "reports",
        exist_ok=True
    )

    plt.figure()

    shap.summary_plot(
    shap_values,
    X,
    show=False
)

    plt.savefig(
        "reports/shap_summary.png",
        bbox_inches='tight'
    )

    plt.close()

    print("✅ SHAP summary plot saved")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🧠 EXPLAINABLE AI ENGINE")
    print("=" * 60)

    model_path = (
        "models/trained_models/purchase_model.pkl"
    )

    data_path = (
        "data/processed/features.csv"
    )

    # Load model
    model = load_model(model_path)

    # Load dataset
    df = load_dataset(data_path)

    # Prepare features
    X = prepare_features(df)

    print("\nColumns Used For SHAP:")
    print(X.columns.tolist())

    print("\nDataset Shape:")
    print(X.shape)

    # Generate SHAP values
    shap_values = generate_shap_values(
        model,
        X
    )

    # Plot SHAP summary
    plot_shap_summary(
        shap_values,
        X
    )

    print(
        "\n🎉 Explainability pipeline completed!"
    )