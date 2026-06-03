import pandas as pd
import os


def load_bias_data(path):
    """
    Load behavioral bias dataset
    """

    df = pd.read_csv(path)

    print("✅ Bias analysis dataset loaded")

    return df


def generate_business_insights(df):

    insights = []

    # Cart abandonment analysis
    abandonment_rate = (
        df['cart_abandonment']
        .mean()
    )

    if abandonment_rate > 0.10:

        insights.append(
            "Users frequently abandon carts before purchase. "
            "Consider simplifying checkout flow."
        )

    # Comparison overload
    overload_rate = (
        df['comparison_overload']
        .mean()
    )

    if overload_rate > 0.05:

        insights.append(
            "Users compare many products before deciding. "
            "Improve recommendation precision."
        )

    # Decision fatigue
    fatigue_rate = (
        df['decision_fatigue']
        .mean()
    )

    if fatigue_rate > 0.15:

        insights.append(
            "Long browsing sessions suggest decision fatigue "
            "among users."
        )

    # Segment analysis
    segment_counts = (
        df['user_segment']
        .value_counts()
    )

    top_segment = (
        segment_counts.idxmax()
    )

    insights.append(
        f"Most users belong to "
        f"'{top_segment}' segment."
    )

    return insights

def display_insights(insights):

    print("\n" + "=" * 50)
    print("📈 BUSINESS INSIGHTS")
    print("=" * 50)

    for i, insight in enumerate(insights, start=1):

        print(f"\n{i}. {insight}")


def save_insights(insights, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(output_path, "w") as file:

        for insight in insights:

            file.write(insight + "\n")

    print("\n✅ Business insights saved")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("📊 BUSINESS INSIGHT ENGINE")
    print("=" * 60)

    input_path = (
        "data/processed/bias_analysis.csv"
    )

    output_path = (
        "data/processed/business_insights.txt"
    )

    # Load dataset
    df = load_bias_data(input_path)

    # Generate insights
    insights = generate_business_insights(df)

    # Display insights
    display_insights(insights)

    # Save insights
    save_insights(
        insights,
        output_path
    )

    print(
        "\n🎉 Business insight generation completed!"
    )