import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame):
    """Build preprocessing pipeline (fitted only on training data)."""

    # 1. Drop identifier-like columns (if present)
    drop_cols = ["Transaction_ID", "Customer_ID", "IP_Address", "Product_SKU", "Store_ID"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # 2. Identify categorical & numerical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("Fraud_Flag").tolist()

    # 3. Define transformations with imputation
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fill NaNs with most common value
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # fill NaNs with median
        ("scaler", StandardScaler())
    ])

    # 4. ColumnTransformer applies the right pipeline to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_cols),
            ("num", numeric_transformer, num_cols),
        ]
    )

    return preprocessor, cat_cols, num_cols

