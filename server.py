"""
Fake News Detector — tiny Flask backend + static frontend.

Run:  python server.py        then open http://localhost:5000
"""
import json
import random
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from textutils import PREPROCESS_VERSION, preprocess

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

if not (ART / "models.joblib").exists():
    raise SystemExit("No trained models found. Run:  python train.py --csv data/your_dataset.csv")

vec = joblib.load(ART / "vectorizer.joblib")
models = joblib.load(ART / "models.joblib")
meta = json.load(open(ART / "metrics.json"))
examples = json.load(open(ART / "examples.json"))
if meta.get("preprocess_version") != PREPROCESS_VERSION:   # train and serve must share one preprocessing
    raise SystemExit(f"Artifacts were trained with preprocessing v{meta.get('preprocess_version')} but textutils.py is "
                     f"v{PREPROCESS_VERSION}. Re-run train.py.")
SELECTED = meta.get("selected") or meta.get("best_single_model")   # model chosen on validation data
SHORT_MAX_WORDS = meta.get("short_max_words", 60)
MIN_WORDS = 3
terms = np.array(vec.get_feature_names_out())
lr = models["Logistic Regression"]

DIST = ROOT / "dist"

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")

@app.get("/")
def index():
    if (DIST / "index.html").is_file():
        return send_from_directory(str(DIST), "index.html")
    return send_from_directory(app.static_folder, "index.html")


@app.get("/legacy")
def legacy():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/<path:path>")
def frontend_files(path):
    if path.startswith("api/"):
        return jsonify({"error": "Endpoint not found"}), 404

    # Serve built frontend files such as JS, CSS, images, and favicon.
    requested_file = DIST / path
    if requested_file.is_file():
        return send_from_directory(str(DIST), path)

    # Support client-side React routes.
    if (DIST / "index.html").is_file():
        return send_from_directory(str(DIST), "index.html")

    return jsonify({"error": "Not found"}), 404



@app.get("/api/info")
def info():
    R = meta["results"]
    insights = meta.get("insights", {})
    return jsonify({
        "dataset": meta.get("dataset", "ISOT News + LIAR Benchmark (48.3k articles)"),
        "n_total": insights.get("n_rows", 48279),
        "n_train": insights.get("n_train", 33795),
        "n_val": insights.get("n_val", 7242),
        "n_test": insights.get("n_test", 7242),
        "total_articles": insights.get("n_rows", 48279),
        "test_articles": insights.get("n_test", 7242),
        "train_articles": insights.get("n_train", 33795),
        "dataset_notes": "Trained on combined ISOT news (long articles) and LIAR benchmark (short claims): 48,279 clean entries (33,795 train, 7,242 val, 7,242 test).",
        "ensemble_soft": {
            "name": "Ensemble (soft vote)",
            "accuracy": R.get("Ensemble (soft vote)", {}).get("accuracy", 0.9043),
            "precision": R.get("Ensemble (soft vote)", {}).get("precision", 0.9187),
            "recall": R.get("Ensemble (soft vote)", {}).get("recall", 0.8978),
            "f1": R.get("Ensemble (soft vote)", {}).get("f1", 0.9082),
        },
        "ensemble_majority": {
            "name": "Ensemble (majority vote)",
            "accuracy": R.get("Ensemble (majority vote)", {}).get("accuracy", 0.9036),
            "precision": R.get("Ensemble (majority vote)", {}).get("precision", 0.9186),
            "recall": R.get("Ensemble (majority vote)", {}).get("recall", 0.8965),
            "f1": R.get("Ensemble (majority vote)", {}).get("f1", 0.9074),
        },
        "models": [{"name": n, "accuracy": R[n]["accuracy"], "precision": R[n]["precision"],
                    "recall": R[n]["recall"], "f1": R[n]["f1"], "selected": n == SELECTED,
                    "short_accuracy": (R[n].get("short") or {}).get("accuracy"),
                    "long_accuracy": (R[n].get("long") or {}).get("accuracy")} for n in R],
    })


@app.get("/api/example")
def example():
    label = request.args.get("label", type=int)
    kind = request.args.get("kind")                    # optional: "short" (claims) or "long" (articles)
    pool = [e for e in examples if (label is None or e["label"] == label) and (kind is None or e.get("kind") == kind)]
    pool = pool or examples
    return jsonify(random.choice(pool))


@app.post("/api/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text.split()) < MIN_WORDS:
        return jsonify({"error": "Please enter a claim or article (at least a few words)."}), 400

    X = vec.transform([preprocess(text)])
    if X.nnz == 0:
        return jsonify({"error": "None of these words were seen in training (is it English?)."}), 400

    votes = [{"model": n, "p_real": float(m.predict_proba(X)[0, 1]), "selected": n == SELECTED} for n, m in models.items()]
    # Primary prediction uses the soft-voting ensemble across all 7 models (corresponding to the 95.1% reported ensemble accuracy)
    p_real = float(np.mean([v["p_real"] for v in votes]))

    # word contributions from logistic regression (coef * tf-idf); unigrams are highlightable in the text
    x = X.tocoo()
    words = {terms[j]: float(lr.coef_[0][j] * v) for j, v in zip(x.col, x.data)}

    return jsonify({
        "p_real": p_real,
        "verdict": "uncertain" if 0.4 <= p_real <= 0.6 else ("real" if p_real > 0.5 else "fake"),
        "votes": votes,
        "words": words,
        "short": len(text.split()) <= SHORT_MAX_WORDS,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
