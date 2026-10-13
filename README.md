# Get Legit — Fake News Detection

An end-to-end machine learning project that analyzes news text and predicts whether it resembles examples labeled **real** or **fake** in its training data. Get Legit combines natural language processing, multiple classical machine learning classifiers, ensemble evaluation, and a web-based interface.

> **Important limitation:** Get Legit is an educational text-classification system, not a definitive fact-checker. It does not independently verify claims against authoritative sources. Predictions can be wrong, particularly for short claims that lack context.

## Features

- Binary classification of news text as real or fake.
- Combined dataset containing long-form news articles and short political claims.
- Text preprocessing and TF-IDF feature extraction.
- Seven classical machine learning classifiers.
- Soft-voting and majority-voting ensemble evaluation.
- Model performance metrics, including accuracy, precision, recall, and F1 score.
- Flask API for inference and model information.
- Web interface for submitting text and exploring model performance.
- Saved model and vectorizer artifacts for inference without retraining.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Text representation | TF-IDF |
| Model persistence | joblib |
| Backend API | Flask |
| Frontend | React, TypeScript, Vite |
| Styling | Tailwind CSS |

## Datasets

The project combines two sources with different text characteristics.

**1. ISOT Fake News Dataset**

Contains longer news articles labeled as fake or real. It provides article-level text for training and evaluation.

**2. LIAR Benchmark Dataset**

Contains short political claims with six original truthfulness ratings. For this project's binary classification task, the labels are mapped as follows:

| Original label | Binary mapping |
|---|---|
| `pants-fire` | Fake |
| `false` | Fake |
| `barely-true` | Fake |
| `half-true` | Excluded |
| `mostly-true` | Real |
| `true` | Real |

This mapping simplifies a multi-level fact-checking task into binary classification. It also introduces limitations: a claim labeled `barely-true` may contain some accurate information, and the original ratings are not perfectly equivalent to binary truth.

### Dataset preparation

The data pipeline performs cleaning and normalization, handles duplicate and conflicting texts, removes unsuitable short or empty examples, and combines the retained records into a unified dataset.

According to the latest project audit, the resulting dataset contains **48,279 cleaned examples**. The reported split is:

| Subset | Examples | Purpose |
|---|---:|---|
| Training | 33,795 | Fit the vectorizer and train classifiers |
| Validation | 7,242 | Model selection and hyperparameter tuning |
| Held-out test | 7,242 | Final performance evaluation |
| **Total** | **48,279** | |

The split counts and dataset size should be regenerated from the actual data pipeline if the dataset changes.

## Machine Learning Workflow

1. **Data ingestion:** Load the ISOT and LIAR datasets.
2. **Label normalization:** Convert the retained examples to binary real/fake labels.
3. **Text preprocessing:** Normalize text and remove unsuitable records, duplicates, and conflicting examples.
4. **Dataset splitting:** Create training, validation, and held-out test subsets.
5. **Feature extraction:** Fit a TF-IDF vectorizer on the training data and transform the remaining subsets using the fitted vectorizer.
6. **Model training:** Train seven classical machine learning classifiers.
7. **Model selection:** Use validation results to compare models and configurations.
8. **Evaluation:** Calculate performance metrics on held-out test data.
9. **Ensemble evaluation:** Compare soft-voting and majority-voting strategies.
10. **Deployment:** Save the trained artifacts and load them in the Flask application for predictions.

### TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) converts text into numerical features that machine learning algorithms can process.

It gives greater relative importance to terms that are informative for a document and less importance to terms that appear throughout many documents.

The vectorizer is fitted on training text only to reduce the risk of data leakage.

## Models

The project evaluates seven classifiers:

1. Logistic Regression
2. Linear Support Vector Machine (SVM)
3. Passive Aggressive Classifier
4. Multinomial Naive Bayes
5. Random Forest
6. Decision Tree
7. K-Nearest Neighbors (KNN)

The algorithms learn different decision boundaries and make different assumptions about the data. Comparing them helps identify their relative strengths and weaknesses.

### Ensemble strategies

**Soft voting:** Combines the class probabilities produced by the participating classifiers and selects the class with the higher averaged probability.

**Majority voting:** Combines the discrete predictions of the classifiers and selects the class receiving the most votes. With seven binary classifiers, four votes are sufficient for a majority.

An ensemble is not guaranteed to outperform every individual classifier. Its performance must be measured on held-out data.

## Evaluation Metrics

The project uses four common classification metrics.

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of positive predictions that are correct.
- **Recall:** The proportion of actual positive examples correctly identified.
- **F1 score:** The harmonic mean of precision and recall.

These metrics provide complementary information. Accuracy alone can conceal differences in false positives and false negatives.

## Latest Reported Evaluation

The latest evaluation report records the following results on a held-out test set of 7,242 examples.

| Model | Accuracy |
|---|---:|
| Linear SVM | 90.57% |
| Logistic Regression | 90.43% |
| Random Forest | 90.02% |
| Passive Aggressive | 89.78% |
| Multinomial Naive Bayes | 87.82% |
| Decision Tree | 86.85% |
| KNN | 84.13% |
| Soft-voting ensemble | 90.43% |
| Majority-voting ensemble | 90.36% |

These figures are the results reported by the latest project audit and should be kept synchronized with `artifacts/metrics.json`.

### An important evaluation caveat

Performance differs substantially by text length in the reported evaluation:

- Long-form news articles: approximately 91–99% accuracy, depending on the classifier.
- Short political claims: approximately 60–65% accuracy, depending on the classifier.

The overall accuracy is influenced by the composition of the test set. Strong results on long articles should not be interpreted as equally strong performance on short claims.

Potential reasons include limited context, differences in label definitions, dataset-source characteristics, and the difficulty of verifying nuanced statements from text alone. Further testing on independently sourced data is necessary to assess generalization.

## Backend API

The Flask backend exposes endpoints for the web application.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the application interface |
| `/api/info` | GET | Returns dataset and model evaluation information |
| `/api/example` | GET | Returns an example text |
| `/api/predict` | POST | Predicts the class of submitted text |

### Prediction request

Send a JSON request to `/api/predict`:

```json
{
  "text": "Insert a news article or claim here."
}
```

The API validates the input, preprocesses the text, transforms it using the saved TF-IDF vectorizer, and obtains model predictions. The response contains the prediction, probability information, model votes, and other supported analysis fields.

Inputs that are too short or contain no features recognized by the fitted vectorizer may be rejected.

## Model Artifacts

The training pipeline saves artifacts used by the inference application. These may include:

- `artifacts/models.joblib` — trained classifiers.
- `artifacts/vectorizer.joblib` — fitted TF-IDF vectorizer.
- `artifacts/metrics.json` — evaluation results and dataset metadata.
- `artifacts/examples.json` — sample inputs for the interface.

The training and serving pipelines must use compatible preprocessing logic and matching artifacts. After changing the data, preprocessing, or models, regenerate the relevant artifacts and rebuild the frontend if required.

## Running the Project

Run these commands from the directory containing the corresponding project files. If the repository's backend and frontend have been moved into separate folders, first navigate to the appropriate directory.

### 1. Set up Python

```bash
python -m venv .venv
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install frontend dependencies

From the directory containing `package.json`:

```bash
npm install
```

### 3. Build the frontend

```bash
npm run build
```

### 4. Start the backend

From the directory containing `server.py`:

```bash
python server.py
```

Open the local address printed by the Flask server. In the current configuration, the application uses `http://127.0.0.1:5000`.

**Note:** Training is separate from running the application. If model artifacts already exist and match the current preprocessing version, retraining is not necessary just to launch the interface. If retraining is required, consult the current `train.py` command-line options and dataset requirements.

## Limitations and Future Improvements

- Improve performance on short political claims.
- Evaluate on independent datasets from sources not used during training.
- Investigate dataset and publisher-related biases.
- Perform error analysis by text length, source, and class.
- Explore more robust language representations and contextual NLP models.
- Improve probability calibration and uncertainty handling.
- Add evidence retrieval and source verification for stronger fact-checking capabilities.

## Responsible Use

The application provides a model-based prediction, not proof that a statement is true or false. Its output should support human review rather than replace independent source verification.

## Project Purpose

Get Legit demonstrates an end-to-end machine learning workflow, from dataset preparation and feature extraction to model evaluation, API integration, and an interactive frontend. It is intended for educational use and further experimentation in natural language processing and misinformation detection.
