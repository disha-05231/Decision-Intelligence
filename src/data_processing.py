import pandas as pd
import os


def load_data(path):
    """
    Load dataset from CSV file
    """
    try:
        df = pd.read_csv(path)
        print("✅ Dataset loaded successfully")
        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def inspect_data(df):
    """
    Display basic dataset information
    """

    print("\n" + "=" * 50)
    print("📊 DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nShape of Dataset: {df.shape}")

    print("\n📌 Columns:")
    print(df.columns.tolist())

    print("\n📌 Data Types:")
    print(df.dtypes)

    print("\n📌 Missing Values:")
    print(df.isnull().sum())

    print("\n📌 First 5 Rows:")
    print(df.head())

    print("\n📌 Duplicate Rows:")
    print(df.duplicated().sum())


def standardize_columns(df):
    """
    Standardize column names
    """

    print("\n🔄 Standardizing column names...")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("✅ Standardized Columns:")
    print(df.columns.tolist())

    return df


def convert_timestamp(df):
    """
    Convert timestamp column to datetime format
    """

    print("\n⏳ Converting timestamp column...")

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print("✅ Timestamp converted successfully")

    except Exception as e:
        print(f"❌ Timestamp conversion failed: {e}")

    return df


def clean_data(df):
    """
    Clean dataset
    """

    print("\n🧹 Cleaning dataset...")

    initial_shape = df.shape

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing essential IDs
    df = df.dropna(subset=['userid', 'sessionid'])

    final_shape = df.shape

    print(f"✅ Removed {initial_shape[0] - final_shape[0]} rows")

    return df


def create_basic_summary(df):
    """
    Create action summary analytics
    """

    print("\n📈 Creating action summary...")

    action_summary = (
        df['eventtype']
        .value_counts()
        .reset_index()
    )

    action_summary.columns = [
        'event_type',
        'count'
    ]

    # Save summary
    output_path = (
        "data/processed/action_summary.csv"
    )

    action_summary.to_csv(
        output_path,
        index=False
    )

    print("✅ Action summary saved")

    print("\n📊 ACTION SUMMARY")
    print(action_summary)

    return action_summary


def save_processed_data(df, output_path):
    """
    Save cleaned dataset
    """

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)

        print(f"\n✅ Processed data saved successfully")
        print(f"📁 Location: {output_path}")

    except Exception as e:
        print(f"❌ Error saving processed data: {e}")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🚀 HUMAN DECISION INTELLIGENCE ENGINE")
    print("📌 DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # File paths
    input_path = "data/raw/clickstream.csv"
    output_path = "data/processed/processed_data.csv"

    # Step 1: Load dataset
    df = load_data(input_path)

    # Stop if loading failed
    if df is not None:

        # Step 2: Inspect dataset
        inspect_data(df)

        # Step 3: Standardize columns
        df = standardize_columns(df)

        # Step 4: Convert timestamps
        df = convert_timestamp(df)

        # Step 5: Clean dataset
        df = clean_data(df)

        # Step 6: Generate summary
        create_basic_summary(df)

        # Step 7: Save processed data
        save_processed_data(df, output_path)

        print("\n🎉 Data preprocessing completed successfully!")

    else:
        print("\n❌ Pipeline stopped due to loading error")