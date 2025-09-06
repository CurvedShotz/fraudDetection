import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocess import load_data, build_preprocessor


def train_models(data_path: str = "data/luxury_cosmetics_fraud_analysis_2025.csv"):
    """Train baseline fraud detection models and save them."""

    # 1. Load data
    print("📂 Loading data...")     #logging prints for the terminal
    df = load_data(data_path)

    # 2. Separate features and target
    
    print("🧹 Preprocessing data...")
    X = df.drop(columns=["Fraud_Flag"])     
    y = df["Fraud_Flag"]

    # 3. Train/test split
    
    print("✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Build preprocessor
    print("⚙️ Building preprocessor...")
    preprocessor, cat_cols, num_cols = build_preprocessor(df)

    # 5. Define models with pipelines
    
    print("🤖 Training Logistic Regression...")
    logreg = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

        
    print("🤖 Training Random Forest...")
    rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])

    # 6. Fit models
    print("✅ Evaluating models...")
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # 7. Evaluate quick scores
    print("✅ Logistic Regression Train Accuracy:", logreg.score(X_train, y_train))
    print("✅ Logistic Regression Test Accuracy:", logreg.score(X_test, y_test))
    print("✅ Random Forest Train Accuracy:", rf.score(X_train, y_train))
    print("✅ Random Forest Test Accuracy:", rf.score(X_test, y_test))

    # 8. Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(logreg, "models/logreg_model.pkl")
    joblib.dump(rf, "models/rf_model.pkl")

    print("📦 Models saved to models/ directory.")


if __name__ == "__main__":
    train_models()
