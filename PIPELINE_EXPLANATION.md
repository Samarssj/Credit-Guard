# CreditGuard: End-to-End Fraud Detection Pipeline

## 1. System overview

CreditGuard is a Streamlit application that takes anonymized credit-card transaction features and returns a fraud probability plus an operational decision. The system has two distinct workflows:

| Workflow | Purpose |
| --- | --- |
| Training workflow | Reads the labeled `creditcard.csv` dataset, preprocesses the data, handles class imbalance, trains the model, evaluates it, and exports artifacts. |
| Inference workflow | Loads the saved model artifact, accepts one transaction or a CSV batch, generates fraud probabilities, applies a configurable threshold, and displays results. |

The separation is important: **training happens offline**, while the deployed Streamlit application performs only fast inference using the already-fitted model.

## 2. Input data

The benchmark dataset contains 30 numeric input features and one label:

| Column group | Meaning |
| --- | --- |
| `Time` | Seconds elapsed since the first transaction in the dataset. |
| `V1`–`V28` | Anonymized PCA-derived transaction features. Their original business meanings are intentionally unavailable. |
| `Amount` | Transaction amount. |
| `Class` | Training label: `0` for legitimate and `1` for fraud. This label is not required when scoring new transactions. |

The benchmark is highly imbalanced: legitimate transactions greatly outnumber fraudulent transactions. Therefore, a model that predicts every row as legitimate could achieve high accuracy while being useless for fraud detection.

## 3. Training pipeline

The training process is implemented in `train_model.py`.

### Step 1: Load and validate

The script loads `data/creditcard.csv`, verifies that all required columns are present, confirms that the target contains both `0` and `1`, and separates the features from the `Class` label.

```text
X = [Time, V1, ..., V28, Amount]
y = Class
```

### Step 2: Stratified train/test split

The data is split into training and test partitions using a stratified split. Stratification preserves approximately the same fraud proportion in both partitions.

```text
Training data: 80%
Test data:     20%
Random seed:   42
```

The test data is kept untouched until final evaluation. This is essential because it represents the data distribution the model has not seen during fitting.

### Step 3: Preprocess numerical features

`Time` and `Amount` are passed through `RobustScaler`. Robust scaling uses medians and interquartile ranges, making it less sensitive to extreme transaction values than ordinary standardization.

The PCA-derived `V1`–`V28` features are passed through unchanged because the benchmark already supplies them in transformed numerical form.

Conceptually:

```text
Time, Amount  -> RobustScaler
V1 ... V28    -> passthrough
```

The scaler is fitted only on the training data inside the pipeline. The test data is transformed using the training-derived parameters, which prevents preprocessing leakage.

### Step 4: Handle class imbalance with SMOTE

SMOTE means **Synthetic Minority Over-sampling Technique**. It creates synthetic examples of the minority class using neighboring fraud examples.

In this project, SMOTE is placed inside an imbalanced-learn pipeline after preprocessing. It is applied only to the training data. The default target ratio is `0.25`, meaning the fraud class is increased to approximately 25% of the majority-class size during training. The original notebook explored stronger 50/50 balancing; the lower default ratio keeps training faster while still exposing the model to substantially more fraud examples.

The test set is never SMOTE-resampled. Resampling the test set would produce an unrealistic evaluation distribution and could lead to misleading results.

### Step 5: Compare models and select the best estimator

The training script compares five candidate classifiers inside the same preprocessing and SMOTE pipeline: logistic regression, random forest, histogram gradient boosting, ExtraTrees, and XGBoost. `GridSearchCV` evaluates the candidates with stratified cross-validation and uses **average precision** as the selection metric. This is better aligned with the rare-fraud setting than selecting by accuracy.

The selected `GridSearchCV.best_estimator_` is then cloned and refit on the complete training split. It is saved as `models/fraud_pipeline.joblib`, so the deployed application does not need to know which candidate won. The current trained artifact selected `HistGradientBoostingClassifier`. This is the best model within the tested candidate set and validation design, not a guarantee of theoretical maximum performance.

Any selected classifier that implements `predict_proba` produces a continuous score that is converted into a probability-like fraud score:

```text
fraud_probability = model.predict_proba(transaction)[:, 1]
```

The model is packaged together with preprocessing and SMOTE as one artifact:

```text
models/fraud_pipeline.joblib
```

This prevents the deployed app from accidentally using a different scaler or feature order than the training process.

## 4. Evaluation pipeline

After training, the model predicts probabilities for the untouched test set. The system calculates:

| Metric | What it measures |
| --- | --- |
| Accuracy | Overall fraction of correct predictions; potentially misleading under extreme imbalance. |
| Precision | Of the transactions flagged as fraud, how many are actually fraud. |
| Recall | Of the actual fraud transactions, how many the model catches. |
| F1 score | Harmonic mean of precision and recall. |
| ROC-AUC | Ranking quality across classification thresholds. |
| Average precision | Area under the precision-recall curve; especially informative for rare positive classes. |
| Confusion matrix | Counts of true negatives, false positives, false negatives, and true positives. |

The saved evaluation artifacts are:

```text
models/metrics.json
models/test_predictions.csv
```

The Streamlit evaluation page loads these files instead of retraining the model every time the app starts.

## 5. Inference workflow in Streamlit

### Single-transaction prediction

On the **Single transaction** page, the user enters `Time`, `Amount`, and `V1`–`V28`. The app constructs a one-row DataFrame in the exact feature order expected by the model:

```text
[Time, V1, V2, ..., V28, Amount]
```

The saved pipeline then performs the following automatically:

```text
Input row
  -> RobustScaler for Time and Amount
  -> passthrough V1–V28
  -> fitted best-estimator classifier
  -> fraud probability
```

### Threshold decision

The probability is compared with the sidebar threshold:

```text
if fraud_probability >= threshold:
    decision = "Review / likely fraud"
else:
    decision = "Likely legitimate"
```

The default threshold is `0.50`, but the user can adjust it. A lower threshold generally increases recall and catches more fraud, but it also increases false positives and manual review workload. A higher threshold generally reduces false positives but may miss more fraud.

The app displays the probability, threshold, decision, and a gauge visualization. The score is a model output, not a guaranteed real-world probability or an automatic financial decision.

### Batch scoring

On the **Batch scoring** page, the user uploads a CSV containing the 30 feature columns. The app validates that all required features are present and numeric, applies the saved pipeline to every row, and appends:

```text
fraud_probability
decision
```

The user can download the scored CSV. If the uploaded file contains `Class`, that column is preserved for comparison but is not used during prediction.

## 6. Streamlit application architecture

The application is organized into four user-facing pages:

| Page | Function |
| --- | --- |
| Overview | Shows dataset size, fraud rate, model configuration, class distribution, and headline metrics. |
| Single transaction | Screens one manually entered transaction. |
| Batch scoring | Scores an uploaded CSV and provides a downloadable result. |
| Model evaluation | Explores threshold-sensitive precision, recall, F1, confusion matrix, ROC curve, and precision-recall curve. |

The model is loaded using Streamlit resource caching. This means the `joblib` artifact is loaded once and reused rather than reloaded on every widget interaction.

## 7. Deployment pipeline

For Streamlit Community Cloud, the repository needs the following:

```text
app.py
requirements.txt
models/fraud_pipeline.joblib
models/metadata.json
models/metrics.json
models/test_predictions.csv
```

At startup, Streamlit installs the packages in `requirements.txt`, launches `app.py`, loads the model and metadata from the `models/` directory, and exposes the interactive interface.

The raw training dataset is not required for normal inference and is intentionally excluded from version control. It is needed only when retraining locally:

```bash
python train_model.py --data data/creditcard.csv --output-dir models
streamlit run app.py
```

## 8. End-to-end summary

```text
Public labeled dataset
        |
        v
Validate schema and labels
        |
        v
Stratified train/test split
        |
        +------------------------------+
        |                              |
        v                              v
Training partition                Untouched test partition
        |                              |
        v                              |
Robust-scale Time and Amount          Wait for final evaluation
        |
        v
Apply SMOTE only to training data
        |
        v
Compare candidate models with GridSearchCV
        |
        v
Select best_estimator_ by average precision
        |
        v
Refit winner and export fraud_pipeline.joblib
        |
        v
Streamlit loads artifact
        |
        v
Single row or uploaded CSV
        |
        v
Fraud probability
        |
        v
Compare with user-selected threshold
        |
        v
Likely legitimate OR Review / likely fraud
```

The central design principle is **separation of concerns**: the training script creates a reproducible model artifact containing whichever estimator won validation, and the Streamlit app focuses on reliable model-agnostic inference, threshold exploration, visualization, and user interaction.
