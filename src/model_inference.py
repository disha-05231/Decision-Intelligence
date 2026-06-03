import pandas as pd
import joblib


def load_model(model_path):
    """
    Load trained ML model
    """

    model = joblib.load(model_path)

    print("✅ Trained model loaded")

    return model


def create_sample_user():
    """
    Create sample user behavior
    """

    sample_data = {
        'session_duration': [1800],
        'click_frequency': [12],
        'product_switch_count': [5],
        'cart_abandonment': [0]
    }

    df = pd.DataFrame(sample_data)

    return df


def predict_purchase(model, user_data):
    """
    Predict purchase behavior
    """

    prediction = model.predict(user_data)[0]

    probability = (
        model.predict_proba(user_data)[0][1]
    )

    return prediction, probability


def display_result(prediction, probability):

    print("\n" + "=" * 50)
    print("🧠 PURCHASE PREDICTION RESULT")
    print("=" * 50)

    if prediction == 1:
        print("✅ User likely to PURCHASE")
    else:
        print("❌ User likely to ABANDON")

    print(
        f"\n📊 Purchase Probability: "
        f"{probability:.2%}"
    )


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🤖 REAL-TIME MODEL INFERENCE")
    print("=" * 60)

    model_path = (
        "models/trained_models/purchase_model.pkl"
    )

    # Load trained model
    model = load_model(model_path)

    # Create sample user
    user_data = create_sample_user()

    print("\n📌 Sample User Data:")
    print(user_data)

    # Predict behavior
    prediction, probability = (
        predict_purchase(
            model,
            user_data
        )
    )

    # Display result
    display_result(
        prediction,
        probability
    )

    print(
        "\n🎉 Model inference completed!"
    )