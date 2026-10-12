# Fake News Detector (DSE 2270 PBL-2)

Detects whether a **short claim** or a **full article** reads as real or fake. Flask backend + one static HTML page.
Trained on news articles (ISOT and/or the "Fake or Real News" CSV) plus the **LIAR** short-claim dataset.

## Commands

```bash
pip install -r requirements.txt

# 1. get LIAR (downloads + verifies 12,836 statements into data/liar/)
python fetch_liar.py

# 2. news articles: put ONE OR BOTH of these in data/
#      data/Fake.csv + data/True.csv        (ISOT / Kaggle "Fake and Real News")
#      data/fake_or_real_news.csv           (title,text,label)

# 3. train  (≈4 min; writes artifacts/ which server.py reads)
python train.py --csv data/fake_or_real_news.csv --liar-dir data/liar
#   with ISOT:      python train.py --fake data/Fake.csv --true data/True.csv --liar-dir data/liar
#   with both:      python train.py --csv data/fake_or_real_news.csv --fake data/Fake.csv --true data/True.csv --liar-dir data/liar
#   add --ablate-preprocessing to also (re)run the validation-only preprocessing ablation

# 4. run
python server.py            # http://localhost:5000
```

## Pipeline (train.py)
load + label-normalise → de-duplicate / drop label conflicts / clean → stratified 70/15/15 train/val/test split
→ TF-IDF fitted on **train only** → Naive Bayes, Logistic Regression, Linear SVM over a hyper-parameter grid and
`min_df` ∈ {1, 2, 3} → selection on **validation** → one final evaluation on the untouched **test** set.

* **Selection rule:** validation score = 0.7 × macro-F1 on short claims + 0.3 × macro-F1 on long articles. No ensemble: the
  single best model decides the verdict (`p_real` in the API). The other two are returned in `votes` for comparison only.
* **Short vs long:** ≤ 60 raw words = short claim, otherwise long article. Metrics (accuracy, precision, recall, F1 with REAL
  as positive class, macro-F1, ROC-AUC, confusion matrix `[[TN,FP],[FN,TP]]`) are stored for overall / short / long in
  `artifacts/metrics.json`, together with the majority-class baseline.
* **Leakage controls:** exact duplicates removed and texts carrying conflicting labels dropped *before* splitting (LIAR's own
  split has 11 duplicate groups spanning train/valid/test, so it is re-split); vectoriser fitted on train only;
  hyper-parameters chosen on validation only; test touched once; LIAR speaker/party/context/credit-history columns are not used.

## LIAR → binary label mapping
LIAR uses PolitiFact's six ratings. Default (`--liar-mapping drop-half`):

| LIAR label | binary | reason |
|---|---|---|
| pants-fire, false | FAKE (0) | inaccurate |
| barely-true | FAKE (0) | PolitiFact: contains an element of truth but ignores critical facts that give a different impression — misleading overall |
| **half-true** | **dropped** | the genuine middle ("partially accurate"); forcing it to either side would add label noise |
| mostly-true, true | REAL (1) | accurate (mostly-true "needs clarification") |

Alternatives for comparison: `--liar-mapping half-real` (half-true → REAL) and `half-fake` (half-true → FAKE).
Consequence to be aware of: at inference the app must still answer for half-true-style claims — the 40–60 % "uncertain" band handles some of these.

## Preprocessing (textutils.py, v2 — one function, `preprocess`, used by train and server)
Negations kept (no, nor, not, never, none, nobody, nothing, nowhere, neither, cannot, without; `n't` expanded), a few
quantifiers kept (all, any, few, more, most, only, than, against), numbers become class tokens (`_pct_`, `_money_`, `_year_`,
`_num_`) instead of being deleted, `!`/`?` become tokens. `PREPROCESS_VERSION` is saved in `metrics.json`; `server.py`
refuses to start if it differs from `textutils.py`. Validation-only ablation (`artifacts/preprocessing_ablation.json`,
Logistic Regression, short / long macro-F1): v1 0.632 / 0.936 → +negations 0.629 / 0.933 → +quantifiers 0.640 / 0.936
→ +numbers 0.646 / 0.933 → +`!?` 0.652 / 0.929. Negation alone is not a measurable win here (it is within noise); the gain on short claims comes mainly from numbers, quantifiers and punctuation.

## Results (held-out test, 2,430 texts: 1,550 short / 880 long; selected = Logistic Regression C=10, min_df=1)
| | accuracy | short claims | long articles |
|---|---|---|---|
| majority-class baseline | 53.8 % | 55.7 % | 50.3 % |
| Naive Bayes | 70.7 % | 58.9 % | 91.4 % |
| **Logistic Regression (selected)** | 73.1 % | 61.0 % | 94.5 % |
| SVM (linear) | 73.4 % | 61.2 % | 94.8 % |

Short-claim ROC-AUC is ≈ 0.63: fact-checked political claims are hard to judge from wording alone, and these numbers are in the
range usually reported for text-only models on LIAR. The previous McIntire-only model scored 53.5 % on the same LIAR test
claims (below the 55.5 % majority baseline); the new model scores 60.3 % on them. Logistic Regression and the linear SVM, and the three `min_df`
values, are within about 0.004 of each other on validation, so that choice is not a strong one (Naive Bayes is clearly behind).
The short slice is 98 % LIAR claims; the long slice contains only news articles.

## Files
- `train.py` – pipeline above · `textutils.py` – preprocessing · `fetch_liar.py` – download + verify LIAR
- `server.py` – Flask API (`/api/predict`, `/api/info`, `/api/example[?label=0|1&kind=short|long]`) · `static/index.html` – frontend
- `artifacts/` – `models.joblib`, `vectorizer.joblib`, `metrics.json`, `examples.json`, `preprocessing_ablation.json`

The system scores writing style and vocabulary; it does not verify facts.
