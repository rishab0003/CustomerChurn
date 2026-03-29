# Customer Churn Prediction System

An end-to-end machine learning project for customer churn prediction with training, validation, batch scoring, and an interactive Streamlit dashboard.

## Overview

This project includes:
- A churn model training pipeline (`src/model_training.py`)
- A prediction engine for batch and single-record inference (`src/prediction.py`)
- A modern multi-page Streamlit app (`app/app.py`)
- Model validation tooling with pass/fail thresholds (`src/test_model.py`)
- Deployment assets (`Dockerfile`, `docker-compose.yml`, `run.sh`)

The current app is churn-focused in UI and outputs (for example: `churn_prediction`, `churn_probability`, `churn_risk_level`).

## Current Capabilities

### 1) Predict Churn (Batch CSV)
- Upload a CSV file and score all rows in one run
- Auto-aligns columns to model schema
- Auto-fills missing required fields with safe defaults
- Returns downloadable predictions CSV with churn-specific columns

### 2) Analytics
- Dataset-level exploratory visualizations
- Uploaded prediction analytics (from the latest scoring run)
- Class distribution and churn probability distribution charts

### 3) Model Info
- Feature importance visualization
- Real evaluation metrics computed from labeled data:
  - Accuracy
  - F1-score
  - ROC-AUC
- Confusion matrix and ROC curve visualizations

## Project Structure

```text
customer_churn_prediction/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── customer_data.csv
│   └── processed/
├── models/
│   ├── churn_model_best.pkl
│   └── churn_model_best_preprocessing.pkl
├── src/
│   ├── data_loading.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── test_model.py
│   └── visualization_eda.py
├── notebooks/
│   └── 01_eda_analysis.py
├── results/
│   └── plots/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
└── README.md
```

## Setup

### Prerequisites
- Python 3.10+ (3.11 recommended)
- `pip`

For Streamlit Cloud deployments, this repo includes `runtime.txt` pinned to Python 3.11 for consistent dependency resolution.

### Installation

```bash
cd customer_churn_prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/app.py
```

Open: `http://localhost:8501`

If port 8501 is busy:

```bash
streamlit run app/app.py --server.port 8502
```

## End-to-End Workflow

### Step 1: Train (or retrain) the model

Default churn training:

```bash
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
```

Generic binary target training (optional):

```bash
python src/model_training.py --data-path path/to/data.csv --target-col target_column --positive-label positive_value
```

### Step 2: Validate model quality

```bash
python src/test_model.py \
  --data-path data/raw/customer_data.csv \
  --model-path models/churn_model_best.pkl \
  --preprocessing-path models/churn_model_best_preprocessing.pkl
```

Optional thresholds:

```bash
python src/test_model.py --data-path data/raw/customer_data.csv --min-auc 0.75 --min-accuracy 0.70
```

### Step 3: Score customer data

In the app:
1. Go to **Predict Churn**
2. Download template CSV
3. Fill rows with customer data
4. Upload CSV
5. Click **Predict All Rows**
6. Download prediction results

### Step 4: Analyze predictions

Go to **Analytics** to see:
- Uploaded prediction summary cards
- Churn class distribution (pie)
- Churn probability histogram
- Preview of predicted churn columns

## Prediction Output Schema

The prediction pipeline returns both generic and churn-friendly fields.

Primary fields used by UI:
- `churn_prediction`
- `churn_probability`
- `no_churn_probability`
- `churn_risk_level`

Compatibility fields also included:
- `prediction`
- `positive_probability`
- `negative_probability`
- `risk_level`

## Model Approach

Training pipeline compares:
- Logistic Regression
- Random Forest
- Gradient Boosting

Best model is selected by ROC-AUC and saved with preprocessing metadata.

Preprocessing includes:
- Feature schema tracking
- Label encoders for categorical features
- Standard scaling for numeric features
- Numeric default values for robust inference

## Troubleshooting

### 1) Model not found

Train the model artifacts first:

```bash
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
```

### 2) Streamlit port already in use

```bash
streamlit run app/app.py --server.port 8502
```

### 3) Feature name warnings from sklearn

Use the latest code in this repo. The prediction path already converts feature matrices correctly for inference.

### 4) Uploaded CSV has missing columns

The app auto-fills required missing fields and shows which columns were filled.

### 5) Error installing requirements on Streamlit Cloud

If you see "Error installing requirements", ensure the app is using the repository `runtime.txt` (Python 3.11) and redeploy.

## Deployment

### Docker

Build and run:

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

### Docker Compose

```bash
docker-compose up --build
```

## Key Files

- App: `app/app.py`
- Training: `src/model_training.py`
- Prediction: `src/prediction.py`
- Validation: `src/test_model.py`
- Data loading: `src/data_loading.py`

## Notes

- Predictions are probabilistic, not guarantees.
- Always combine model outputs with business/domain judgment.
- Validate model quality on representative data before production decisions.

---

**Version:** 2.0  
**Last Updated:** 29 March 2026  
**Status:** Ready for training, scoring, analytics, and validation
# CustomerChurn
