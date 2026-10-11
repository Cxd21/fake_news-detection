# Fake News Detector (DSE 2270 PBL-2)

Seven classifiers (Naive Bayes, Logistic Regression, Passive-Aggressive, linear SVM, Random Forest,
Decision Tree, KNN) vote on whether an article is real or fake. Plain Flask backend + one static HTML page.

## Run
    pip install -r requirements.txt
    python server.py            # open http://localhost:5000

`artifacts/` already contains trained models, so it works immediately.

## Retrain (e.g. on ISOT from your report)
    python train.py --fake data/Fake.csv --true data/True.csv
    # or: python train.py --csv data/some_dataset.csv     (title/text/label columns)

## Files
- server.py: Flask API (`/api/predict`, `/api/info`, `/api/example`) + serves the page
- static/index.html: the whole frontend (no framework, no build step)
- train.py: dedup, clean, TF-IDF, split, train 7 models, held-out evaluation, save to artifacts/
- textutils.py: preprocessing shared by training and serving
