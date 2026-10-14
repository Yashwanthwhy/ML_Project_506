"""
Train Wine Quality Random Forest Model and save to wine_rf_model.joblib
"""
import io, urllib.request, warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
GOOD_THRESHOLD = 6
MODEL_PATH = "wine_rf_model.joblib"

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol", "is_white"
]

def main():
    print("Downloading dataset...")
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red = pd.read_csv(red_url, sep=";")
    white = pd.read_csv(white_url, sep=";")

    red["wine_type"] = "Red"
    white["wine_type"] = "White"
    df = pd.concat([red, white], ignore_index=True).drop_duplicates().dropna().reset_index(drop=True)
    print(f"Dataset cleaned: {len(df):,} rows")

    df["is_white"] = (df["wine_type"] == "White").astype(int)
    df["good_quality"] = (df["quality"] >= GOOD_THRESHOLD).astype(int)

    X = df[FEATURES]
    y = df["good_quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    print("Training Random Forest...")
    # Best parameters for this wine quality benchmark
    best_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1
    )
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Evaluation Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    payload = {
        "model": best_rf,
        "features": FEATURES,
        "good_threshold": GOOD_THRESHOLD,
        "feature_importances": dict(zip(FEATURES, best_rf.feature_importances_)),
        "best_params": best_rf.get_params(),
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
