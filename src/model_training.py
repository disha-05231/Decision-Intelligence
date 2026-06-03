import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import json
from imblearn.over_sampling import SMOTE
# ======================================
# 1. LOAD DATASET
# ======================================

def load_data(
    path="data/processed/features.csv"
):

    print("\n📥 Loading clustered dataset...")

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"❌ File not found: {path}"
        )

    df = pd.read_csv(path)
    print(df.columns.tolist())

    print(
        f"✅ Dataset loaded successfully | Shape: {df.shape}"
    )

    return df


# ======================================
# 2. CREATE TARGET VARIABLE
# ======================================

def create_target(df):

    print("\n🎯 Creating purchase target...")

    # Purchase target
    # 1 = purchase completed
    # 0 = cart abandoned

    df["purchase_target"] = (
    df["purchase_ratio"] >= 0.15
).astype(int)
    
    print(df["purchase_target"].value_counts())
    print("✅ Target variable created")

    return df


# ======================================
# 3. PREPARE FEATURES
# ======================================

def prepare_features(df):

    print("\n⚙️ Preparing ML features...")

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

    y = df['purchase_target']

    print("\n📊 Feature Sample:")
    print(X.head())
    print("\nPurchase Target Distribution")
    print(df["purchase_target"].value_counts())

    print("\nPurchase Ratio Statistics")
    print(df["purchase_ratio"].describe())
    print("\nFeature Statistics:")
    print(X.describe())

    print("\n📈 Target Distribution:")
    print(y.value_counts())

    return X, y



# ======================================
# 4. SPLIT DATA
# ======================================

def split_data(X, y):

    print("\n✂️ Splitting dataset...")
    
    print(X.columns.tolist())

    return train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================
# 5. TRAIN MODEL
# ======================================

def train_model(X_train, y_train):

    print("\n⚖️ Applying SMOTE...")

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    print("✅ SMOTE completed")

    print("\n🤖 Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    print("✅ Model trained successfully")

    return model


# ======================================
# 6. EVALUATE MODEL
# ======================================

def evaluate_model(

    model,
    X_test,
    y_test

):

    print("\n📊 Evaluating model...")

    os.makedirs(
        "reports",
        exist_ok=True
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
    y_test,
    y_pred
)

    auc_score = roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:,1]
)

    report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)
    
    print(
    f"\n🎯 Accuracy: {accuracy:.4f}"
)

    print(
    f"\n🎯 ROC AUC: {auc_score:.4f}"
)

    print("\n📋 Classification Report:")

    print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

    metrics = {

    "accuracy": float(accuracy),
    "roc_auc": float(auc_score),

    "precision": float(
        report["weighted avg"]["precision"]
    ),

    "recall": float(
        report["weighted avg"]["recall"]
    ),

    "f1": float(
        report["weighted avg"]["f1-score"]
    )
}

    with open(
    "reports/model_metrics.json",
    "w"
) as f:

        json.dump(
        metrics,
        f,
        indent=4
    )

    print("✅ Metrics saved")

    # =========================
# CONFUSION MATRIX
# =========================

    cm = confusion_matrix(
    y_test,
    y_pred
)

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

    disp.plot()

    plt.title("Confusion Matrix")

    plt.savefig(
    "reports/confusion_matrix.png"
)

    plt.close()

    print("✅ Confusion matrix saved")


# =========================
# ROC CURVE
# =========================

    y_prob = model.predict_proba(
    X_test
)[:, 1]

    RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

    plt.title("ROC Curve")

    plt.savefig(
    "reports/roc_curve.png"
)

    plt.close()

    print("✅ ROC curve saved")

    return y_pred


# ======================================
# 7. FEATURE IMPORTANCE
# ======================================

def plot_feature_importance(

    model,
    feature_names

):

    print("\n📊 Generating feature importance plot...")

    os.makedirs(
        "reports",
        exist_ok=True
    )

    importance = model.feature_importances_

    importance_df = pd.DataFrame({

        'feature': feature_names,
        'importance': importance

    })

    importance_df = (

        importance_df
        .sort_values(by='importance')

    )

    plt.figure(figsize=(8, 5))

    plt.barh(

        importance_df['feature'],
        importance_df['importance']

    )

    plt.xlabel("Importance Score")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png"
    )

    plt.close()

    print("✅ Feature importance plot saved")


# ======================================
# 8. SAVE MODEL
# ======================================

def save_model(model):

    print("\n💾 Saving trained model...")

    os.makedirs(

        "models/trained_models",
        exist_ok=True

    )

    joblib.dump(

        model,
        "models/trained_models/purchase_model.pkl"

    )

    print("✅ Model saved successfully")


# ======================================
# 9. COMPLETE PIPELINE
# ======================================

def run_pipeline():

    # Load dataset
    df = load_data()

    # Create target
    df = create_target(df)

    # Prepare features
    X, y = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    # Evaluate model
    evaluate_model(

        model,
        X_test,
        y_test

    )

    # Feature importance
    plot_feature_importance(

        model,
        X.columns

    )

    # Save trained model
    save_model(model)


# ======================================
# 10. MAIN EXECUTION
# ======================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("🤖 PURCHASE PREDICTION MODEL")

    print("=" * 60)

    try:

        run_pipeline()

        print(
            "\n🎉 Model pipeline completed successfully!"
        )

    except Exception as e:

        print(
            f"\n❌ ERROR: {e}"
        )