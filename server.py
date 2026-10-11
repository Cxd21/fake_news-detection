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

from textutils import clean_text, scrub_source_artifacts

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

if not (ART / "models.joblib").exists():
    raise SystemExit("No trained models found. Run:  python train.py --csv data/your_dataset.csv")

vec = joblib.load(ART / "vectorizer.joblib")
models = joblib.load(ART / "models.joblib")
meta = json.load(open(ART / "metrics.json"))
examples = json.load(open(ART / "examples.json"))
terms = np.array(vec.get_feature_names_out())
lr = models["Logistic Regression"]

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    R = meta["results"]
    return jsonify({
        "dataset": meta["dataset"],
        "n_test": meta["insights"]["n_test"],
        "models": [{"name": n, "accuracy": R[n]["accuracy"], "precision": R[n]["precision"],
                    "recall": R[n]["recall"], "f1": R[n]["f1"]} for n in R],
    })


@app.get("/api/example")
def example():
    label = request.args.get("label", type=int)
    pool = [e for e in examples if label is None or e["label"] == label]
    return jsonify(random.choice(pool))


@app.post("/api/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text.split()) < 8:
        return jsonify({"error": "Please paste at least a few sentences."}), 400

    X = vec.transform([clean_text(scrub_source_artifacts(text))])
    if X.nnz == 0:
        return jsonify({"error": "None of these words were seen in training (is it English?)."}), 400

    votes = [{"model": n, "p_real": float(m.predict_proba(X)[0, 1])} for n, m in models.items()]
    p_real = float(np.mean([v["p_real"] for v in votes]))

    # word contributions from logistic regression (coef * tf-idf); unigrams are highlightable in the text
    x = X.tocoo()
    words = {terms[j]: float(lr.coef_[0][j] * v) for j, v in zip(x.col, x.data)}

    return jsonify({
        "p_real": p_real,
        "verdict": "uncertain" if 0.4 <= p_real <= 0.6 else ("real" if p_real > 0.5 else "fake"),
        "votes": votes,
        "words": words,
        "short": len(text.split()) < 40,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
