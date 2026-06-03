import pandas as pd
import os


def load_clustered_data(path):
    """
    Load clustered behavioral dataset
    """

    df = pd.read_csv(path)

    print("✅ Clustered dataset loaded")

    return df


def detect_loss_aversion(df):
    """
    Users add to cart but abandon
    """

    df['loss_aversion'] = (
        (df['cart_abandonment'] == 1)
        &
        (df['click_frequency'] > 5)
    ).astype(int)

    return df


def detect_comparison_overload(df):
    """
    Users compare too many products
    """

    df['comparison_overload'] = (
        df['product_switch_count'] > 5
    ).astype(int)

    return df


def detect_decision_fatigue(df):
    """
    Long sessions with abandonment
    """

    df['decision_fatigue'] = (
        (df['session_duration'] > 1800)
        &
        (df['cart_abandonment'] == 1)
    ).astype(int)

    return df


def detect_impulsive_behavior(df):
    """
    Fast purchase tendency
    """

    df['impulsive_behavior'] = (
        (df['click_frequency'] < 5)
        &
        (df['cart_abandonment'] == 0)
    ).astype(int)

    return df


def generate_bias_summary(df):

    print("\n" + "=" * 50)
    print("🧠 BIAS ANALYSIS SUMMARY")
    print("=" * 50)

    biases = [
        'loss_aversion',
        'comparison_overload',
        'decision_fatigue',
        'impulsive_behavior'
    ]

    summary = {}

    for bias in biases:

        count = df[bias].sum()

        summary[bias] = count

        print(f"{bias}: {count}")

    return summary


def generate_text_insights(summary):

    insights = []

    # Loss aversion
    if summary['loss_aversion'] > 500:

        insights.append(
            "Users frequently abandon after adding products to cart."
        )

    # Comparison overload
    if summary['comparison_overload'] > 300:

        insights.append(
            "Users compare too many products before making decisions."
        )

    # Decision fatigue
    if summary['decision_fatigue'] > 500:

        insights.append(
            "Long browsing sessions may be causing decision fatigue."
        )

    # Impulsive behavior
    if summary['impulsive_behavior'] > 50:

        insights.append(
            "A subset of users exhibit impulsive buying behavior."
        )

    return insights

def save_bias_analysis(df, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print("✅ Bias analysis saved")


def save_text_insights(insights, output_path):

    with open(output_path, "w") as file:

        for insight in insights:
            file.write(insight + "\n")

    print("✅ Bias insights text saved")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🧠 BEHAVIORAL BIAS DETECTION")
    print("=" * 60)

    input_path = (
        "data/processed/clustered_data.csv"
    )

    output_csv = (
        "data/processed/bias_analysis.csv"
    )

    output_txt = (
        "data/processed/bias_insights.txt"
    )

    # Load dataset
    df = load_clustered_data(input_path)

    # Detect biases
    df = detect_loss_aversion(df)

    df = detect_comparison_overload(df)

    df = detect_decision_fatigue(df)

    df = detect_impulsive_behavior(df)

    # Generate summary
    summary = generate_bias_summary(df)

    # Generate insights
    insights = generate_text_insights(summary)

    # Save outputs
    save_bias_analysis(df, output_csv)

    save_text_insights(insights, output_txt)

    print(
        "\n🎉 Bias detection completed successfully!"
    )