"""
Training pipeline for the Fake News Detection System.

Pipeline (matches the project report / PPT):
  data collection -> cleaning -> TF-IDF -> train/test split
  -> multiple classifiers -> evaluation -> ensemble -> save artefacts

Usage
-----
# ISOT / Kaggle "Fake and Real News" (Fake.csv + True.csv in ./data)
python train.py --fake data/Fake.csv --true data/True.csv

# or a single CSV with a text column and a label column (FAKE/REAL or 0/1)
python train.py --csv data/fake_or_real_news.csv

Labels follow the report: REAL = 1, FAKE = 0.
All metrics are computed on a held-out test set the models never saw.
"""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from textutils import clean_text, scrub_source_artifacts

SEED = 42
OUT = Path("artifacts")


def load_data(args) -> pd.DataFrame:
    if args.fake and args.true:
        fake = pd.read_csv(args.fake)
        true = pd.read_csv(args.true)
        fake["label"], true["label"] = 0, 1
        df = pd.concat([fake, true], ignore_index=True)
    elif args.csv:
        df = pd.read_csv(args.csv)
        lab = df[args.label_col]
        if pd.api.types.is_numeric_dtype(lab):
            df["label"] = lab
        else:
            df["label"] = lab.astype(str).str.strip().str.upper().map({"REAL": 1, "TRUE": 1, "FAKE": 0, "FALSE": 0})
    else:
        raise SystemExit("Provide --csv, or both --fake and --true")

    title = df["title"].fillna("") if "title" in df else ""
    df["raw"] = (title + ". " + df[args.text_col].fillna("")).astype(str)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[["raw", "label"]]


def metrics(y, pred, proba):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # [[TN,FP],[FN,TP]]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--fake")
    ap.add_argument("--true")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--name", help="dataset name shown in the app")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    # ---------- 1. data ----------
    df = load_data(args)
    n_raw = len(df)
    df = df.drop_duplicates(subset="raw").reset_index(drop=True)   # duplicates would leak across the split
    df["raw"] = df["raw"].map(scrub_source_artifacts)               # remove publisher give-aways
    df["clean"] = df["raw"].map(clean_text)
    df = df[df["clean"].str.split().str.len() >= 20].reset_index(drop=True)  # drop empty / stub rows
    print(f"rows: {n_raw} raw -> {len(df)} after dedup / cleaning")

    # ---------- 2. split ----------
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"], test_size=args.test_size, stratify=df["label"], random_state=SEED)

    # ---------- 3. features (fit on TRAIN only) ----------
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)
    print("tf-idf:", Xtr.shape)

    # ---------- 4. models (the algorithms named in the report) ----------
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000, random_state=SEED),
        "Passive Aggressive": CalibratedClassifierCV(SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
                                                              max_iter=1000, random_state=SEED), cv=3),  # = Passive-Aggressive I
        "SVM (Linear)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=40, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance", algorithm="brute"),
    }

    results, test_proba = {}, {}
    for name, m in models.items():
        s = time.time()
        m.fit(Xtr, y_tr)
        proba = m.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        r = metrics(y_te, pred, proba)
        r["train_seconds"] = round(time.time() - s, 1)
        results[name] = r
        test_proba[name] = proba
        fpr, tpr, _ = roc_curve(y_te, proba)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int)
        r["roc"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
        print(f"{name:22s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f} ({r['train_seconds']}s)")

    # ---------- 5. ensemble (soft voting + hard majority voting) ----------
    P = np.column_stack(list(test_proba.values()))
    soft = P.mean(axis=1)
    r = metrics(y_te, (soft >= 0.5).astype(int), soft)
    fpr, tpr, _ = roc_curve(y_te, soft)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int)
    r["roc"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    results["Ensemble (soft vote)"] = r
    hard = (np.mean(P >= 0.5, axis=1) > 0.5).astype(int)
    hr = metrics(y_te, hard, hard)
    hr.pop("roc_auc")
    results["Ensemble (majority vote)"] = hr
    print(f"{'Ensemble (soft)':22s} acc={results['Ensemble (soft vote)']['accuracy']:.4f}")
    print(f"{'Ensemble (majority)':22s} acc={hr['accuracy']:.4f}")

    # ---------- 6. explainability + data insights ----------
    lr = models["Logistic Regression"]
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    top_fake = [(terms[i], float(lr.coef_[0][i])) for i in order[:25]]
    top_real = [(terms[i], float(lr.coef_[0][i])) for i in order[::-1][:25]]

    raw_te = df.loc[X_te.index, "raw"]
    insights = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "class_counts": {"FAKE": int((df.label == 0).sum()), "REAL": int((df.label == 1).sum())},
        "avg_words": {"FAKE": float(df[df.label == 0].clean.str.split().str.len().mean()),
                      "REAL": float(df[df.label == 1].clean.str.split().str.len().mean())},
        "top_fake_terms": top_fake, "top_real_terms": top_real,
        "tfidf_features": int(Xtr.shape[1]),
        "caps_ratio": {"FAKE": float(raw_te[y_te == 0].map(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)).mean()),
                       "REAL": float(raw_te[y_te == 1].map(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)).mean())},
        "exclaim_per_1k": {"FAKE": float(raw_te[y_te == 0].map(lambda t: 1000 * t.count("!") / max(len(t), 1)).mean()),
                           "REAL": float(raw_te[y_te == 1].map(lambda t: 1000 * t.count("!") / max(len(t), 1)).mean())},
    }

    # ---------- 7. save ----------
    best = max((k for k in results if "Ensemble" not in k), key=lambda k: results[k]["f1"])
    joblib.dump(vec, OUT / "vectorizer.joblib", compress=3)
    joblib.dump(models, OUT / "models.joblib", compress=3)
    json.dump({"results": results, "insights": insights, "best_single_model": best,
               "test_size": args.test_size, "seed": SEED,
               "dataset": args.name or Path(args.csv or args.fake).stem,
               "trained_at": time.strftime("%Y-%m-%d %H:%M")},
              open(OUT / "metrics.json", "w"), indent=1)
    # a few unseen test articles for the app's "try an example" buttons
    ex = pd.DataFrame({"text": raw_te, "label": y_te})
    ex["len"] = ex.text.str.len()
    ex = ex[(ex.len > 400) & (ex.len < 2500)]
    pd.concat([ex[ex.label == 0].sample(5, random_state=SEED),
               ex[ex.label == 1].sample(5, random_state=SEED)])[["text", "label"]].to_json(
        OUT / "examples.json", orient="records")
    print(f"best single model: {best}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
