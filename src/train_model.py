"""
train_model.py

Trains a logistic regression xG model on the shot features we built,
evaluates it against a held-out test set, and compares our predicted
probabilities to StatsBomb's own professional xG values.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import joblib

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "shots.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "xg_model.pkl"

NUMERIC_FEATURES = ["distance_to_goal", "angle_to_goal"]
CATEGORICAL_FEATURES = ["body_part", "shot_type", "under_pressure"]
TARGET = "is_goal"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Simplify body_part into three buckets: Head, Foot, Other.
    df["body_part"] = df["body_part"].apply(
        lambda b: "Head" if b == "Head" else ("Foot" if "Foot" in str(b) else "Other")
    )
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(max_iter=1000)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    pred_proba = pipe.predict_proba(X_test)[:, 1]

    print("=== Model performance on held-out test set ===")
    print(f"ROC AUC:      {roc_auc_score(y_test, pred_proba):.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"Log loss:     {log_loss(y_test, pred_proba):.3f}  (lower is better)")
    print(f"Brier score:  {brier_score_loss(y_test, pred_proba):.3f}  (lower is better)")

    print()
    print("=== Comparison to StatsBomb's own professional xG model ===")
    sb_xg = df_test["statsbomb_xg"].fillna(0).values
    print(f"Our model  - avg predicted xG: {pred_proba.mean():.3f} | actual goal rate: {y_test.mean():.3f}")
    print(f"StatsBomb  - avg predicted xG: {sb_xg.mean():.3f} | actual goal rate: {y_test.mean():.3f}")
    corr = np.corrcoef(pred_proba, sb_xg)[0, 1]
    print(f"Correlation between our xG and StatsBomb's xG: {corr:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
