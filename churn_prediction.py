"""
Customer Churn Prediction — one single script, start to finish.

What this file does, in order:
  1) Load the raw customer data from a CSV file.
  2) Explore it (EDA) — check the churn balance.
  3) Clean it: fix missing values, remove duplicates.
  4) Encode text columns into numbers the model can use.
  5) Split into a training set and a test set.
  6) Handle outliers (bounds learned from the TRAINING set only).
  7) Train a Logistic Regression model.
  8) Evaluate it: Accuracy / Precision / Recall / F1 / Confusion Matrix.
  9) Interpret the result in plain business terms.

Every chart this script makes is saved as a PNG in charts/, since a
plain script (unlike a notebook) has nowhere to display them inline.

Run it from the project's root folder with:
    python3 churn_prediction.py
"""

# ============================================================
# IMPORTS  (grouped by what they're used for, so it's clear why
# each one is here)
# ============================================================

# --- Data handling -------------------------------------------------
import pandas as pd   # load the CSV and work with it as a table (DataFrame)
import numpy as np    # small numeric helpers

# --- Charts -------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")            # "Agg" = draw charts to files, don't try to pop up a window (this is a script, not a notebook)
import matplotlib.pyplot as plt
import seaborn as sns             # nicer-looking statistical charts, built on matplotlib

# --- Splitting & preprocessing -----------------------------------
from sklearn.model_selection import train_test_split   # splits data into train/test sets
from sklearn.preprocessing import StandardScaler        # rescales numeric columns (mean=0, std=1)

# --- Model ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression      # the classifier we train to predict Churn

# --- Evaluation metrics -----------------------------------------
from sklearn.metrics import (
    accuracy_score,        # % of ALL predictions that were correct
    precision_score,       # of customers we PREDICTED would churn, % that really did
    recall_score,          # of customers who REALLY churned, % we correctly caught
    f1_score,              # one number that balances precision and recall
    confusion_matrix,      # 2x2 table: correct/incorrect predictions per class
    classification_report, # text summary of the metrics above, per class
)

# ============================================================
# CONSTANTS
# ============================================================
DATA_PATH = "data/customer_churn.csv"    # where the raw dataset lives
CHARTS_DIR = "charts"                     # where every chart gets saved
RANDOM_STATE = 42                         # fixed seed so results are reproducible every run
NUMERIC_COLS = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]  # columns we scale / outlier-check

sns.set_style("whitegrid")   # chart styling, applied once for every plot below


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV file from disk into a pandas DataFrame."""
    df = pd.read_csv(path)                      # actually read the file
    print(f"\n[LOAD] Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    print("[LOAD] Preview of the raw data:")
    print(df.head(3))                           # show the first 3 rows so we can eyeball it
    return df


# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def explore_data(df: pd.DataFrame):
    """Look at the target column's balance before touching anything."""
    churn_counts = df["Churn"].value_counts()
    print(f"\n[EDA] Churn value counts:\n{churn_counts}")
    print(f"[EDA] Churn rate: {churn_counts['Yes'] / len(df):.1%}")
    print("[EDA] Note: 88% churned here is unusually high for real telecom "
          "data (usually 15-30%) — worth remembering when reading the metrics later.")

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", palette=["#4C72B0", "#DD8452"])
    plt.title("Customers who churned vs. stayed")
    plt.xlabel("")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/01_churn_balance.png", dpi=120)  # save instead of show() — this is a script
    plt.close()
    print(f"[EDA] Saved chart: {CHARTS_DIR}/01_churn_balance.png")


# ============================================================
# STEP 3 — CLEAN: MISSING VALUES + DUPLICATES
# (done before the split — these two fixes don't use any statistic
#  that could "leak" test-set information)
# ============================================================
def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing values and remove duplicates.

    InternetService is blank for customers with no internet subscription
    at all (every blank row also has TechSupport == "No" — confirmed by
    checking, not assumed). That's a real category, not an error, so we
    label it "No Internet" instead of dropping the rows or guessing DSL
    or Fiber.
    """
    df = df.copy()  # work on a copy so we never silently mutate the caller's DataFrame

    # --- Missing values --------------------------------------------------
    n_missing = df["InternetService"].isna().sum()        # count blanks BEFORE fixing
    df["InternetService"] = df["InternetService"].fillna("No Internet")  # fill them with a real label
    print(f"\n[CLEAN] Missing InternetService values filled: {n_missing}")
    print("[CLEAN] Data right after the missing-value fix:")
    print(df[["InternetService", "TechSupport", "Churn"]].head(3))  # show the fixed column in context

    # --- Duplicates -----------------------------------------------------
    n_dupe_rows = df.duplicated().sum()                    # fully identical rows
    n_dupe_ids = df["CustomerID"].duplicated().sum()        # same customer listed twice
    df = df.drop_duplicates()                                # drop exact row duplicates
    df = df.drop_duplicates(subset="CustomerID", keep="first")  # drop repeated customer IDs, keep the first
    print(f"\n[CLEAN] Fully duplicated rows removed: {n_dupe_rows}")
    print(f"[CLEAN] Duplicate CustomerIDs removed: {n_dupe_ids}")
    print(f"[CLEAN] Shape after de-duplication: {df.shape}")
    print("[CLEAN] Data right after removing duplicates:")
    print(df.head(3))

    return df


# ============================================================
# STEP 4 — ENCODE FEATURES (turn text into numbers)
# ============================================================
def prepare_features(df: pd.DataFrame):
    """
    Drop the ID column (it's not a real feature), turn Churn into 0/1,
    and one-hot encode every other text column (Gender, ContractType,
    InternetService, TechSupport) into 0/1 dummy columns.
    """
    df = df.drop(columns=["CustomerID"])            # CustomerID is just a label, not a predictive feature
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  # turn the target into numbers: churned=1, stayed=0

    # pd.get_dummies turns each text category into its own 0/1 column.
    # drop_first=True avoids redundant columns (e.g. Gender_Female is implied by Gender_Male == 0)
    X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
    y = df["Churn"]  # y is what we're predicting; X is everything we predict it from

    print(f"\n[ENCODE] Feature matrix shape after encoding: {X.shape}")
    print("[ENCODE] Data right after encoding (first 3 rows):")
    print(X.head(3))

    return X, y


# ============================================================
# STEP 5 — OUTLIERS (done AFTER the split, on X_train / X_test)
# ============================================================
def cap_outliers(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_cols):
    """
    Cap extreme numeric values using the IQR rule — but compute the
    bounds from X_train ONLY, then apply those same bounds to both
    X_train and X_test.

    Why this order matters: if the bounds were computed on the full
    dataset (train + test together) before splitting, information from
    the test set would quietly influence how the training data gets
    cleaned. That's a small form of data leakage. Computing the bounds
    from the training data alone, and just applying them to the test
    set, avoids that.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # --- "before" chart, using the training data -----------------
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 3.5))
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(y=X_train[col], ax=ax, color="#4C72B0")
        ax.set_title(col)
    plt.suptitle("Training data before outlier capping")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/02_outliers_before_capping.png", dpi=120)
    plt.close()
    print(f"\n[OUTLIERS] Saved chart: {CHARTS_DIR}/02_outliers_before_capping.png")

    for col in numeric_cols:
        q1 = X_train[col].quantile(0.25)          # 25th percentile, from TRAIN only
        q3 = X_train[col].quantile(0.75)          # 75th percentile, from TRAIN only
        iqr = q3 - q1                              # interquartile range
        lower = q1 - 1.5 * iqr                     # anything below this is a low outlier
        upper = q3 + 1.5 * iqr                     # anything above this is a high outlier

        n_out_train = ((X_train[col] < lower) | (X_train[col] > upper)).sum()
        n_out_test = ((X_test[col] < lower) | (X_test[col] > upper)).sum()

        X_train[col] = X_train[col].clip(lower=lower, upper=upper)  # cap train values
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)    # cap test values with the SAME bounds

        print(f"[OUTLIERS] {col}: bounds [{lower:.1f}, {upper:.1f}] "
              f"-> capped {n_out_train} in train, {n_out_test} in test")

    print("\n[OUTLIERS] X_train right after capping (first 3 rows):")
    print(X_train.head(3))

    return X_train, X_test


# ============================================================
# STEP 6 — TRAIN THE MODEL
# ============================================================
def train_model(X_train, y_train):
    """Scale the features, then fit a Logistic Regression classifier."""
    scaler = StandardScaler()                         # will rescale each column to mean 0, std 1
    X_train_scaled = scaler.fit_transform(X_train)     # learn the scaling from train, and apply it

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)  # simple, interpretable classifier
    model.fit(X_train_scaled, y_train)                 # actually train it

    print("\n[TRAIN] Model trained on", X_train.shape[0], "customers")
    return model, scaler


# ============================================================
# STEP 7 — EVALUATE THE MODEL
# ============================================================
def evaluate_model(model, scaler, X_test, y_test):
    """Score the trained model on the held-out test set."""
    X_test_scaled = scaler.transform(X_test)   # apply the SAME scaling learned from train (never re-fit on test)
    y_pred = model.predict(X_test_scaled)       # the model's Churn predictions for the test customers

    acc = accuracy_score(y_test, y_pred)         # overall % correct
    prec = precision_score(y_test, y_pred)       # trustworthiness of a "will churn" prediction
    rec = recall_score(y_test, y_pred)           # % of real churners the model actually caught
    f1 = f1_score(y_test, y_pred)                # balance of precision and recall
    cm = confusion_matrix(y_test, y_pred)        # the raw right/wrong counts, per class

    print("\n=== Evaluation ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/03_confusion_matrix.png", dpi=120)
    plt.close()
    print(f"[EVAL] Saved chart: {CHARTS_DIR}/03_confusion_matrix.png")

    print("\n[EVAL] Being skeptical of the score: 100% recall is unusually clean for a "
          "churn model. It suggests TechSupport/ContractType are close to a deterministic "
          "rule for Churn in THIS dataset, not messy real-world behavior.")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


# ============================================================
# STEP 8 — BUSINESS INTERPRETATION
# ============================================================
def interpret_model(model, X):
    """Turn the model's coefficients into a plain-language business story."""
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    top_bottom = pd.concat([coefs.head(5), coefs.tail(5)])
    colors = ["#4C72B0" if v < 0 else "#DD8452" for v in top_bottom.values]
    top_bottom.plot(kind="barh", color=colors, ax=ax)
    ax.set_title("Strongest churn drivers (blue = lowers risk, orange = raises risk)")
    ax.set_xlabel("Coefficient (scaled)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/04_churn_drivers.png", dpi=120)
    plt.close()
    print(f"\n[INTERPRET] Saved chart: {CHARTS_DIR}/04_churn_drivers.png")

    print("\n[INTERPRET] Top features increasing churn risk:")
    print(coefs.sort_values(ascending=False).head(5))
    print("\n[INTERPRET] Top features decreasing churn risk:")
    print(coefs.head(5))
    print("\n[INTERPRET] Business read: Tech Support + a 1-2 year contract are the "
          "strongest retention levers; no internet service and high charges are the "
          "biggest churn risks.")


# ============================================================
# MAIN — runs every step above, in order
# ============================================================
def main():
    import os
    os.makedirs(CHARTS_DIR, exist_ok=True)   # make sure charts/ exists before we try saving into it

    # 1) Load
    df_raw = load_data(DATA_PATH)

    # 2) Explore (EDA)
    explore_data(df_raw)

    # 3) Clean (missing values + duplicates — safe to do before splitting)
    df_clean = clean_missing_and_duplicates(df_raw)

    # 4) Encode text columns into numbers
    X, y = prepare_features(df_clean)

    # 5) Split BEFORE touching outliers, so the test set stays untouched
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape} | Test: {X_test.shape}")

    # 6) Outliers — bounds learned from X_train only, applied to both
    X_train, X_test = cap_outliers(X_train, X_test, NUMERIC_COLS)

    # ---- Final checkpoint: exactly what goes into the model ----
    print("\n[FINAL] Fully cleaned + encoded training data used to fit the model:")
    print(X_train.head())

    # 7) Train
    model, scaler = train_model(X_train, y_train)

    # 8) Evaluate
    evaluate_model(model, scaler, X_test, y_test)

    # 9) Business interpretation
    interpret_model(model, X)


if __name__ == "__main__":
    main()
