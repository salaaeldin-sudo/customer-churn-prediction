"""
Customer Churn Prediction — training pipeline.

Loads the customer churn dataset, cleans it (missing values,
duplicates, outliers), trains a Logistic Regression classifier, and
evaluates it with Accuracy / Precision / Recall / Confusion Matrix.

Run:  python3 src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/customer_churn.csv"
RANDOM_STATE = 42
NUMERIC_COLS = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      1) Missing values — InternetService is blank for customers with
         no internet subscription at all (confirmed: every blank row
         also has TechSupport == "No"). That's a real category, not a
         data error, so we fill it with the label "No Internet"
         instead of dropping or guessing DSL/Fiber.
      2) Duplicates — drop any fully duplicated rows and any repeated
         CustomerID.
      3) Outliers — IQR rule on the numeric columns (Age, Tenure,
         MonthlyCharges, TotalCharges), capped rather than dropped so
         we keep every customer in the training data.
    """
    df = df.copy()

    # --- 1) Missing values ---------------------------------------------
    n_missing = df["InternetService"].isna().sum()
    df["InternetService"] = df["InternetService"].fillna("No Internet")
    print(f"[cleaning] Missing InternetService values filled: {n_missing}")

    # --- 2) Duplicates ----------------------------------------------------
    n_dupe_rows = df.duplicated().sum()
    n_dupe_ids = df["CustomerID"].duplicated().sum()
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="CustomerID", keep="first")
    print(f"[cleaning] Fully duplicated rows removed: {n_dupe_rows}")
    print(f"[cleaning] Duplicate CustomerIDs removed: {n_dupe_ids}")

    # --- 3) Outliers (IQR capping) -----------------------------------
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[cleaning] Outliers capped in {col}: {n_out} (bounds {lower:.1f} to {upper:.1f})")

    return df


def prepare_features(df: pd.DataFrame):
    df = df.drop(columns=["CustomerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
    y = df["Churn"]
    return X, y


def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def main():
    df_raw = load_data(DATA_PATH)
    print(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"Churn balance:\n{df_raw['Churn'].value_counts()}\n")

    df_clean = clean_data(df_raw)

    X, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test)

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    print("\nTop features increasing churn risk:")
    print(coefs.head(5))
    print("\nTop features decreasing churn risk:")
    print(coefs.tail(5))


if __name__ == "__main__":
    main()
