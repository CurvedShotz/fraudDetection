# import os
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     roc_curve,
# )
# from sklearn.model_selection import train_test_split

# from src.preprocess import load_data, build_preprocessor


# def evaluate_models(data_path: str = "data/luxury_cosmetics_fraud_analysis_2025.csv"):
#     """Evaluate trained fraud detection models."""

#     # 1. Load data
#     df = load_data(data_path)
#     X = df.drop(columns=["Fraud_Flag"])
#     y = df["Fraud_Flag"]

#     # 2. Train/test split (same as in train.py)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     # 3. Load models
#     logreg = joblib.load("models/logreg_model.pkl")
#     rf = joblib.load("models/rf_model.pkl")

#     models = {"Logistic Regression": logreg, "Random Forest": rf}

#     os.makedirs("outputs", exist_ok=True)

#     # 4. Evaluate each model
#     for name, model in models.items():
#         print(f"\n🔎 Evaluating {name}...")

#         y_pred = model.predict(X_test)
#         y_proba = model.predict_proba(X_test)[:, 1]

#         # Print metrics
#         print(classification_report(y_test, y_pred))

#         # Save confusion matrix
#         cm = confusion_matrix(y_test, y_pred)
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
#         plt.title(f"{name} - Confusion Matrix")
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.tight_layout()
#         plt.savefig(f"outputs/{name}_confusion_matrix.png")
#         plt.close()

#         # Save ROC curve
#         fpr, tpr, _ = roc_curve(y_test, y_proba)
#         auc = roc_auc_score(y_test, y_proba)
#         plt.figure(figsize=(6, 4))
#         plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
#         plt.plot([0, 1], [0, 1], "k--")
#         plt.title(f"{name} - ROC Curve")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.legend(loc="lower right")
#         plt.tight_layout()
#         plt.savefig(f"outputs/{name}_roc_curve.png")
#         plt.close()

#         print(f"✅ Results saved to outputs/ for {name}")


# if __name__ == "__main__":
#     evaluate_models()


### SECOND ITERATION

# src/evaluate.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, build_preprocessor


def evaluate_models(data_path: str = "data/luxury_cosmetics_fraud_analysis_2025.csv"):
    os.makedirs("outputs", exist_ok=True)

    print("📂 Loading data...")
    df = load_data(data_path)
    X = df.drop(columns=["Fraud_Flag"])
    y = df["Fraud_Flag"].astype(int)

    print("✂️ Train/Test split (same seed as train.py)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("⚙️ Loading preprocessor and transforming TEST...")
    preprocessor = joblib.load("models/preprocessor.pkl")
    X_test_enc = preprocessor.transform(X_test)

    try:
        from scipy import sparse
        if sparse.issparse(X_test_enc):
            X_test_enc = X_test_enc.toarray()
    except Exception:
        pass

    # Load any models that exist
    model_files = [f for f in os.listdir("models") if f.endswith("_model.pkl")]
    if not model_files:
        raise FileNotFoundError("No saved models found in ./models")

    for mfile in model_files:
        name = mfile.replace("_model.pkl", "").replace("_", " ").title()  # nice label
        print(f"\n🔎 Evaluating {name}...")
        model = joblib.load(os.path.join("models", mfile))

        # Predict
        y_pred = model.predict(X_test_enc)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_enc)[:, 1]
        else:
            # Some models may not expose predict_proba; fall back if needed
            try:
                y_proba = model.decision_function(X_test_enc)
            except Exception:
                y_proba = None

        # Text report
        report = classification_report(y_test, y_pred, digits=3, target_names=["legit","fraud"])
        print(report)

        # Save text report
        with open(os.path.join("outputs", f"{name}_classification_report.txt"), "w") as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"outputs/{name}_confusion_matrix.png")
        plt.close()

        # ROC + PR only if we have probabilities
        if y_proba is not None:
            # ROC
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title(f"{name} - ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"outputs/{name}_roc_curve.png")
            plt.close()

            # Precision–Recall
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(recall, precision, label=f"AP = {ap:.3f}")
            plt.title(f"{name} - Precision–Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(f"outputs/{name}_pr_curve.png")
            plt.close()

            print(f"📈 AUC: {auc:.3f} | AP: {ap:.3f}")

    print("\n✅ All results saved in ./outputs")


if __name__ == "__main__":
    evaluate_models()
