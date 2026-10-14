"""
Wine Quality Predictor - Flask Backend
Auto-downloads UCI data and trains Random Forest on first run.
"""

import os, io, logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
CORS(app)

MODEL_PATH = os.path.join(BASE_DIR, "wine_rf_model.joblib")
SEED = 42
GOOD_THRESHOLD = 6

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol", "is_white"
]

FEATURE_META = {
    "fixed acidity":        {"min": 4.0,  "max": 16.0,  "step": 0.1,   "unit": "g/L",   "default": 7.4},
    "volatile acidity":     {"min": 0.08, "max": 1.6,   "step": 0.01,  "unit": "g/L",   "default": 0.5},
    "citric acid":          {"min": 0.0,  "max": 1.0,   "step": 0.01,  "unit": "g/L",   "default": 0.3},
    "residual sugar":       {"min": 0.5,  "max": 30.0,  "step": 0.1,   "unit": "g/L",   "default": 3.0},
    "chlorides":            {"min": 0.01, "max": 0.2,   "step": 0.001, "unit": "g/L",   "default": 0.05},
    "free sulfur dioxide":  {"min": 1.0,  "max": 70.0,  "step": 1.0,   "unit": "mg/L",  "default": 25.0},
    "total sulfur dioxide": {"min": 6.0,  "max": 300.0, "step": 1.0,   "unit": "mg/L",  "default": 100.0},
    "density":              {"min": 0.990,"max": 1.004,  "step": 0.0001,"unit": "g/cm3", "default": 0.996},
    "pH":                   {"min": 2.8,  "max": 4.0,   "step": 0.01,  "unit": "",      "default": 3.3},
    "sulphates":            {"min": 0.2,  "max": 2.0,   "step": 0.01,  "unit": "g/L",   "default": 0.6},
    "alcohol":              {"min": 8.0,  "max": 15.0,  "step": 0.1,   "unit": "%",     "default": 10.5},
}


def download_and_train():
    import urllib.request, warnings
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
    warnings.filterwarnings("ignore")

    log.info("Downloading UCI wine quality dataset...")
    red_url   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red   = pd.read_csv(io.StringIO(urllib.request.urlopen(red_url).read().decode()),   sep=";")
    white = pd.read_csv(io.StringIO(urllib.request.urlopen(white_url).read().decode()), sep=";")

    red["wine_type"]   = "Red"
    white["wine_type"] = "White"
    df = pd.concat([red, white], ignore_index=True).drop_duplicates().dropna().reset_index(drop=True)
    log.info(f"Dataset: {len(df):,} rows after cleaning")

    df["is_white"]     = (df["wine_type"] == "White").astype(int)
    df["good_quality"] = (df["quality"] >= GOOD_THRESHOLD).astype(int)

    X = df[FEATURES]
    y = df["good_quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    log.info("Training Random Forest (20-iter RandomizedSearchCV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    param_dist = {
        "n_estimators":     [200, 300, 500],
        "max_depth":        [None, 15, 20, 30],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        param_dist, n_iter=20, scoring="f1", cv=cv,
        random_state=SEED, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    log.info(f"Done | Acc={accuracy_score(y_test,y_pred):.3f} F1={f1_score(y_test,y_pred):.3f} AUC={roc_auc_score(y_test,y_prob):.3f}")

    payload = {
        "model":                best_rf,
        "features":             FEATURES,
        "good_threshold":       GOOD_THRESHOLD,
        "feature_importances":  dict(zip(FEATURES, best_rf.feature_importances_)),
        "best_params":          search.best_params_,
    }
    joblib.dump(payload, MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")
    return payload


if os.path.exists(MODEL_PATH):
    log.info("Loading saved model...")
    MODEL_PAYLOAD = joblib.load(MODEL_PATH)
else:
    MODEL_PAYLOAD = download_and_train()

MODEL    = MODEL_PAYLOAD["model"]
FEAT_IMP = MODEL_PAYLOAD.get("feature_importances", {})


@app.route("/")
def index():
    return render_template("index.html", feature_meta=FEATURE_META, feature_importances=FEAT_IMP)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        values = [float(data.get(f, FEATURE_META.get(f, {}).get("default", 0))) for f in FEATURES]
        df_in  = pd.DataFrame([values], columns=FEATURES)
        pred   = int(MODEL.predict(df_in)[0])
        prob   = float(MODEL.predict_proba(df_in)[0][1])
        label  = "GOOD" if pred == 1 else "POOR"
        conf   = prob if pred == 1 else 1 - prob
        return jsonify({
            "prediction":       label,
            "probability_good": round(prob * 100, 1),
            "confidence":       round(conf * 100, 1),
            "is_good":          pred == 1
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/feature-importance")
def feature_importance():
    fi = sorted(FEAT_IMP.items(), key=lambda x: x[1], reverse=True)
    return jsonify([{"feature": k, "importance": round(v, 4)} for k, v in fi])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
