# Backend — Fake News Detection API & Machine Learning Pipeline

Python Flask REST API and scikit-learn machine learning pipeline.

## Directory Structure

- `server.py` — Flask application exposing `/api/predict`, `/api/info`, and `/api/example`, while serving the built Vite UI from `../dist/`.
- `train.py` — Complete ML training pipeline (data ingestion, cleaning, deduplication, TF-IDF vectorization, model training, grid evaluation, and metric logging).
- `textutils.py` — Preprocessing utilities (URL removal, contraction expansion, number/punctuation tokenization, stop-words, and publisher artifact scrubbing).
- `fetch_liar.py` — Utility script to download the LIAR benchmark dataset into `data/liar/`.
- `requirements.txt` — Python dependencies (`flask`, `scikit-learn`, `pandas`, `numpy`, `joblib`).
- `artifacts/` — Serialized models and evaluation outputs:
  - `models.joblib` — 7 trained classifiers (Linear SVM, Logistic Regression, Random Forest, Passive Aggressive, Naive Bayes, Decision Tree, KNN)
  - `vectorizer.joblib` — TF-IDF vectorizer (50,000 features, sublinear TF)
  - `metrics.json` — Evaluation results, confusion matrices, and split statistics
  - `examples.json` — Sample claims and articles for the interactive analyzer
- `data/` — Dataset storage (e.g. `data/liar/`)
- `static/` — Legacy HTML frontend fallback

## Running the Server

```bash
# From repository root:
.\venv\Scripts\python.exe server.py

# Or from inside backend/:
cd backend
..\venv\Scripts\python.exe server.py
```

## Running Training

```bash
# Example training on combined ISOT + LIAR:
.\venv\Scripts\python.exe backend/train.py --fake C:\Users\rawat\Desktop\pbl\Fake.csv --true C:\Users\rawat\Desktop\pbl\True.csv --liar-dir backend/data/liar --out backend/artifacts --name "ISOT News + LIAR Benchmark (48.3k articles)"
```

