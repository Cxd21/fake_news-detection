"""
Training pipeline for the Fake News Detection System (news articles + LIAR short claims).

    load + label-normalise -> de-duplicate / resolve conflicts + clean -> stratified train/val/test split
    -> TF-IDF fitted on TRAIN only -> compare Naive Bayes / Logistic Regression / Linear SVM
    -> choose TF-IDF min_df, model and hyper-parameters on VALIDATION (short-text weighted)
    -> evaluate once on the untouched TEST set (overall, short claims, long articles)

Labels follow the report: REAL = 1, FAKE = 0.

Data
----
  news articles : --csv data/fake_or_real_news.csv            (title/text/label; FAKE/REAL or 0/1)
                  and/or --fake data/Fake.csv --true data/True.csv   (ISOT / Kaggle "Fake and Real News")
  LIAR claims   : --liar-dir data/liar                         (train.tsv, valid.tsv, test.tsv; python fetch_liar.py)

LIAR label -> binary mapping  (LIAR is rated on PolitiFact's 6-point truth scale)
----------------------------------------------------------------------------------
  default  --liar-mapping drop-half
      pants-fire, false, barely-true  -> FAKE (0)
      mostly-true, true               -> REAL (1)
      half-true                       -> DROPPED (not used for training or evaluation)
  Why: PolitiFact defines "barely true" as containing an element of truth but ignoring critical facts that
  give a different impression, i.e. misleading overall; "mostly true" is accurate but needs clarification.
  "half-true" is by definition the middle ("partially accurate"), so forcing it to either side would inject
  label noise; it is dropped by default. The alternatives are available for comparison:
      half-real : half-true -> REAL      half-fake : half-true -> FAKE
  Note: the app will still have to give a REAL/FAKE answer to half-true-style claims at inference time;
  the 40-60 % "uncertain" band in server.py is the safety valve for those.

Only the claim text is used. LIAR's speaker, party, context and "credit history" columns are NOT features
(they would not exist at inference time, and the credit-history counts can include the claim's own label).
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
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from textutils import PREPROCESS_VERSION, make_cleaner, preprocess, scrub_source_artifacts

SEED = 42
SHORT_MAX_WORDS = 60                    # raw words (title+text); <= this is a "short claim", > is a "long article"
SPLIT = {"val": 0.15, "test": 0.15}     # train = 70 %
MIN_CLEAN_TOKENS = {"news": 20, "liar": 3}      # drop stub rows with (almost) no content after cleaning
SELECT_WEIGHTS = {"short": 0.7, "long": 0.3}    # validation score = weighted macro-F1 (short-text first)
MIN_DF_GRID = (2,)
GRID = {
    "Naive Bayes":         [dict(alpha=0.1)],
    "Logistic Regression": [dict(C=10)],
    "SVM (Linear)":        [dict(C=1.0)],
    "Passive Aggressive":  [dict()],
    "Random Forest":       [dict(n_estimators=100)],
    "Decision Tree":       [dict(max_depth=40)],
    "KNN":                 [dict(n_neighbors=7)],
}
_BASE = {"pants-fire": 0, "false": 0, "barely-true": 0, "mostly-true": 1, "true": 1}
LIAR_MAPPINGS = {"drop-half": dict(_BASE), "half-real": {**_BASE, "half-true": 1}, "half-fake": {**_BASE, "half-true": 0}}
LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
             "c_barely", "c_false", "c_half", "c_mostly", "c_pants", "context"]


# ----------------------------------------------------------------------------- data loading
def load_news(args):
    frames = []
    if args.csv:
        df = pd.read_csv(args.csv)
        lab = df[args.label_col]
        df["label"] = lab if pd.api.types.is_numeric_dtype(lab) else \
            lab.astype(str).str.strip().str.upper().map({"REAL": 1, "TRUE": 1, "FAKE": 0, "FALSE": 0})
        frames.append((Path(args.csv).stem, df))
    if args.fake and args.true:
        fake, true = pd.read_csv(args.fake), pd.read_csv(args.true)
        fake["label"], true["label"] = 0, 1
        frames.append(("ISOT", pd.concat([fake, true], ignore_index=True)))
    out = []
    for name, df in frames:
        title = df["title"].fillna("") if "title" in df else ""
        o = pd.DataFrame({"raw": (title + ". " + df[args.text_col].fillna("")).astype(str), "label": df["label"]})
        o["source"], o["family"] = name, "news"
        out.append(o)
    return out


def load_liar(liar_dir, mapping_name):
    mapping, parts = LIAR_MAPPINGS[mapping_name], []
    for split in ("train", "valid", "test"):
        p = Path(liar_dir) / f"{split}.tsv"
        if not p.exists():
            raise SystemExit(f"{p} not found. Run:  python fetch_liar.py")
        d = pd.read_csv(p, sep="\t", header=None, names=LIAR_COLS, quoting=3, dtype=str, keep_default_na=False)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)           # LIAR's own split is NOT used (see dedup + re-split below)
    counts = d["label"].value_counts().to_dict()
    unknown = set(counts) - set(_BASE) - {"half-true"}
    if unknown:
        raise SystemExit(f"Unexpected LIAR labels {unknown}: wrong file format?")
    d["label_bin"] = d["label"].map(mapping)
    n_dropped = int(d["label_bin"].isna().sum())
    d = d.dropna(subset=["label_bin"])
    o = pd.DataFrame({"raw": d["statement"].astype(str), "label": d["label_bin"].astype(int)})
    o["source"], o["family"] = "LIAR", "liar"
    return o, {"mapping": mapping_name, "mapping_detail": mapping, "raw_label_counts": counts,
               "dropped_unmapped_rows": n_dropped}


def assemble(args):
    stats = {}
    frames = load_news(args)
    if args.liar_dir:
        liar, stats["liar"] = load_liar(args.liar_dir, args.liar_mapping)
        frames.append(liar)
    if not frames:
        raise SystemExit("No data given. Use --csv and/or --fake/--true, and --liar-dir")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    stats["rows_loaded"] = int(len(df))

    # --- de-duplicate; drop every copy of a text that carries conflicting labels
    df["key"] = df["raw"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    conflict = df.groupby("key")["label"].transform("nunique") > 1
    stats["conflicting_label_rows_dropped"] = int(conflict.sum())
    df = df[~conflict]
    before = len(df)
    df = df.drop_duplicates("key")
    stats["duplicate_rows_dropped"] = int(before - len(df))

    # --- clean (identical function to inference), drop near-empty rows
    df = df.copy()
    df["clean"] = df["raw"].map(preprocess)
    ntok = df["clean"].str.split().str.len()
    keep = ntok >= df["family"].map(MIN_CLEAN_TOKENS)
    stats["stub_rows_dropped"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    df["words"] = df["raw"].str.split().str.len()
    df["bucket"] = np.where(df["words"] <= SHORT_MAX_WORDS, "short", "long")
    stats["rows_final"] = int(len(df))
    return df.drop(columns=["key"]), stats


def split(df):
    strat = df["family"] + "_" + df["label"].astype(str)
    tv, te = train_test_split(df.index, test_size=SPLIT["test"], stratify=strat, random_state=SEED)
    tr, va = train_test_split(tv, test_size=SPLIT["val"] / (1 - SPLIT["test"]), stratify=strat[tv], random_state=SEED)
    return df.loc[tr], df.loc[va], df.loc[te]


# ----------------------------------------------------------------------------- models + metrics
def make_model(name, p):
    if name == "Naive Bayes":
        return MultinomialNB(**p)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED, **p)
    if name == "Passive Aggressive":
        return CalibratedClassifierCV(SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, max_iter=1000, random_state=SEED), cv=3)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "KNN":
        return KNeighborsClassifier(metric="cosine", weights="distance", algorithm="brute", n_jobs=-1, **p)
    return CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED, **p), cv=3)


def block(y, proba):
    y = np.asarray(y)
    if len(y) == 0:
        return None
    pred = (proba >= 0.5).astype(int)
    both = len(set(y)) == 2
    return {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),                 # REAL as the positive class
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if both else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),   # [[TN, FP], [FN, TP]]
    }


def sliced(frame, proba):
    y = frame["label"].values
    out = block(y, proba)
    out["short"] = block(y[frame["bucket"].values == "short"], proba[frame["bucket"].values == "short"])
    out["long"] = block(y[frame["bucket"].values == "long"], proba[frame["bucket"].values == "long"])
    return out


def score(sl):
    """Validation selection score: weighted macro-F1 of the short and long slices."""
    s = {k: (sl[k]["macro_f1"] if sl[k] else 0.0) for k in ("short", "long")}
    if sl["short"] is None:                      # no short data at all -> fall back to the long slice
        return s["long"]
    if sl["long"] is None:
        return s["short"]
    return SELECT_WEIGHTS["short"] * s["short"] + SELECT_WEIGHTS["long"] * s["long"]


def new_vectorizer(min_df):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.9, max_features=50000, sublinear_tf=True)


# ----------------------------------------------------------------------------- preprocessing ablation
def ablate_preprocessing(tr, va, out_dir):
    """Cumulative ablation on VALIDATION data (Logistic Regression C=10, min_df=2). Test data is never used."""
    steps = [
        ("v1: letters only, NLTK stop-words (negations dropped, numbers dropped)", dict(keep_negations=False, keep_quantifiers=False, numbers="drop", punct_tokens=False)),
        ("+ keep negations (not/no/never/...)", dict(keep_negations=True, keep_quantifiers=False, numbers="drop", punct_tokens=False)),
        ("+ keep quantifiers (all/any/most/only/...)", dict(keep_negations=True, keep_quantifiers=True, numbers="drop", punct_tokens=False)),
        ("+ numbers as tokens (_pct_/_money_/_year_/_num_)", dict(keep_negations=True, keep_quantifiers=True, numbers="tokens", punct_tokens=False)),
        ("+ ! and ? as tokens  (= v2 production default)", dict(keep_negations=True, keep_quantifiers=True, numbers="tokens", punct_tokens=True)),
    ]
    scrub_tr = tr["raw"].map(scrub_source_artifacts)
    scrub_va = va["raw"].map(scrub_source_artifacts)
    rows = []
    for label, opts in steps:
        cl = make_cleaner(**opts)
        vec = new_vectorizer(2)
        Xtr = vec.fit_transform(scrub_tr.map(cl))
        Xva = vec.transform(scrub_va.map(cl))
        m = make_model("Logistic Regression", {"C": 10}).fit(Xtr, tr["label"])
        sl = sliced(va, m.predict_proba(Xva)[:, 1])
        rows.append({"variant": label, "short_macro_f1": sl["short"]["macro_f1"], "short_accuracy": sl["short"]["accuracy"],
                     "long_macro_f1": sl["long"]["macro_f1"], "long_accuracy": sl["long"]["accuracy"], "score": score(sl)})
        print(f"  {label:78s} short F1m={rows[-1]['short_macro_f1']:.4f}  long F1m={rows[-1]['long_macro_f1']:.4f}")
    json.dump(rows, open(Path(out_dir) / "preprocessing_ablation.json", "w"), indent=1)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="news CSV with title/text/label (e.g. data/fake_or_real_news.csv)")
    ap.add_argument("--fake", help="ISOT Fake.csv")
    ap.add_argument("--true", help="ISOT True.csv")
    ap.add_argument("--liar-dir", help="folder with LIAR train.tsv / valid.tsv / test.tsv")
    ap.add_argument("--liar-mapping", choices=LIAR_MAPPINGS, default="drop-half")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--name", help="dataset name shown in the app")
    ap.add_argument("--out", default="artifacts", help="artefact folder (server.py reads ./artifacts)")
    ap.add_argument("--ablate-preprocessing", action="store_true", help="also run the validation-only preprocessing ablation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()

    # 1-2. load, normalise labels, dedup, clean
    df, data_stats = assemble(args)
    print(f"rows: {data_stats}")
    # 3. stratified train / validation / test split (stratified on source family x label)
    tr, va, te = split(df)
    print(f"split  train {len(tr)}  val {len(va)}  test {len(te)}   (short/long in test: "
          f"{(te.bucket == 'short').sum()}/{(te.bucket == 'long').sum()})")
    ytr = tr["label"].values

    if args.ablate_preprocessing:
        print("\npreprocessing ablation (validation only):")
        ablate_preprocessing(tr, va, out)

    # 4-5. TF-IDF (fit on train only) x models x hyper-parameters, scored on validation
    grid_rows, kept = [], {}
    for md in MIN_DF_GRID:
        vec = new_vectorizer(md)
        Xtr = vec.fit_transform(tr["clean"])
        Xva = vec.transform(va["clean"])
        print(f"\nmin_df={md}: {Xtr.shape[1]:,} features")
        best_for_model = {}
        for name, plist in GRID.items():
            for p in plist:
                s = time.time()
                m = make_model(name, p).fit(Xtr, ytr)
                sl = sliced(va, m.predict_proba(Xva)[:, 1])
                row = {"min_df": md, "model": name, "params": p, "n_features": int(Xtr.shape[1]), "score": score(sl),
                       "val_short_macro_f1": sl["short"]["macro_f1"] if sl["short"] else None,
                       "val_long_macro_f1": sl["long"]["macro_f1"] if sl["long"] else None,
                       "val_accuracy": sl["accuracy"], "val_short_accuracy": sl["short"]["accuracy"] if sl["short"] else None,
                       "val_long_accuracy": sl["long"]["accuracy"] if sl["long"] else None, "val": sl}
                grid_rows.append(row)
                if name not in best_for_model or row["score"] > best_for_model[name][0]["score"]:
                    best_for_model[name] = (row, m)
                print(f"  {name:20s} {str(p):14s} score={row['score']:.4f} short F1m={row['val_short_macro_f1']:.3f} "
                      f"long F1m={row['val_long_macro_f1']:.3f} ({time.time() - s:.0f}s)")
        kept[md] = (vec, best_for_model)

    best = max(grid_rows, key=lambda r: r["score"])
    sel_md, sel_name = best["min_df"], best["model"]
    vec, best_for_model = kept[sel_md]
    print(f"\nSELECTED on validation: {sel_name} {best['params']} with min_df={sel_md}  (score {best['score']:.4f})")

    # 6. final, one-shot evaluation on the untouched test set
    Xte = vec.transform(te["clean"])
    models, results = {}, {}
    for name, (row, m) in best_for_model.items():
        models[name] = m
        proba = m.predict_proba(Xte)[:, 1]
        res = sliced(te, proba)
        res.update(selected=(name == sel_name), params=row["params"], min_df=sel_md, validation_score=row["score"])
        results[name] = res
        print(f"TEST {name:20s} acc={res['accuracy']:.4f}  short acc={res['short']['accuracy'] if res['short'] else float('nan'):.4f}  "
              f"long acc={res['long']['accuracy'] if res['long'] else float('nan'):.4f}")

    # Ensemble (soft voting + hard majority voting)
    test_probas = [m.predict_proba(Xte)[:, 1] for m in models.values()]
    if test_probas:
        P = np.column_stack(test_probas)
        soft = P.mean(axis=1)
        res_soft = sliced(te, soft)
        res_soft.update(selected=False, params={}, min_df=sel_md, validation_score=None)
        results["Ensemble (soft vote)"] = res_soft
        print(f"TEST {'Ensemble (soft)':20s} acc={res_soft['accuracy']:.4f}")

        hard = (np.mean(P >= 0.5, axis=1) > 0.5).astype(int)
        res_hard = sliced(te, hard)
        res_hard.update(selected=False, params={}, min_df=sel_md, validation_score=None)
        results["Ensemble (majority vote)"] = res_hard
        print(f"TEST {'Ensemble (majority)':20s} acc={res_hard['accuracy']:.4f}")

    majority = int(round(ytr.mean()))
    base = sliced(te, np.full(len(te), float(majority)))
    baseline = {"majority_label": majority, **{k: (base[k]["accuracy"] if base[k] else None) for k in ("short", "long")},
                "overall": base["accuracy"]}

    # --- insights (used by the app's info endpoint) + top terms from the linear model
    lr = models["Logistic Regression"]
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    uni = [i for i in order if " " not in terms[i]]
    insights = {
        "n_rows": int(len(df)), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "class_counts": {"FAKE": int((df.label == 0).sum()), "REAL": int((df.label == 1).sum())},
        "by_family_bucket": {f"{f}/{b}": {"n": int(len(g)), "real_share": float(g.label.mean())}
                             for (f, b), g in df.groupby(["family", "bucket"])},
        "test_by_bucket": {b: int((te.bucket == b).sum()) for b in ("short", "long")},
        "avg_words": {b: float(df[df.bucket == b].words.mean()) for b in ("short", "long") if (df.bucket == b).any()},
        "top_fake_terms": [(terms[i], float(lr.coef_[0][i])) for i in uni[:25]],
        "top_real_terms": [(terms[i], float(lr.coef_[0][i])) for i in uni[::-1][:25]],
        "tfidf_features": int(len(terms)),
    }

    dataset = args.name or " + ".join(sorted(set(df["source"]))) + (f" (LIAR mapping: {args.liar_mapping})" if args.liar_dir else "")
    meta = {
        "results": results, "insights": insights, "selected": sel_name,
        "best_single_model": sel_name,                    # key kept for backward compatibility
        "selection": {"criterion": f"validation weighted macro-F1 (short {SELECT_WEIGHTS['short']}, long {SELECT_WEIGHTS['long']})",
                      "min_df": sel_md, "params": best["params"], "score": best["score"]},
        "validation_grid": [{k: v for k, v in r.items() if k != "val"} for r in grid_rows],
        "tfidf_comparison": [  # best validation score per min_df
            {"min_df": md, "n_features": max(r["n_features"] for r in grid_rows if r["min_df"] == md),
             "best_score": max(r["score"] for r in grid_rows if r["min_df"] == md),
             "best_model": max((r for r in grid_rows if r["min_df"] == md), key=lambda r: r["score"])["model"]}
            for md in MIN_DF_GRID],
        "baseline_majority_accuracy_test": baseline,
        "data": data_stats, "short_max_words": SHORT_MAX_WORDS, "preprocess_version": PREPROCESS_VERSION,
        "tfidf": {"ngram_range": [1, 2], "max_df": 0.9, "sublinear_tf": True, "min_df": sel_md},
        "seed": SEED, "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "dataset": dataset, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }

    # 7. save in the paths server.py reads
    if hasattr(vec, "stop_words_"):
        del vec.stop_words_                              # sklearn stores every pruned term here; it only bloats the file
    joblib.dump(vec, out / "vectorizer.joblib", compress=3)
    joblib.dump(models, out / "models.joblib", compress=3)
    json.dump(meta, open(out / "metrics.json", "w"), indent=1)

    # examples drawn from the held-out TEST split: 5 per (kind, label); long ones kept readable in length
    ex = []
    for kind in ("short", "long"):
        pool = te[te.bucket == kind]
        if kind == "long":
            pool = pool[(pool.raw.str.len() > 400) & (pool.raw.str.len() < 2500)]
        for lab in (0, 1):
            g = pool[pool.label == lab]
            ex += [{"text": r.raw, "label": int(lab), "kind": kind, "source": r.source}
                   for r in g.sample(min(5, len(g)), random_state=SEED).itertuples()]
    json.dump(ex, open(out / "examples.json", "w"))
    print(f"\nsaved to {out}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
