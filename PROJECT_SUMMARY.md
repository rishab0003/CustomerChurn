# Project Summary

## Status

Customer churn prediction project is fully functional and ready for local use and container deployment.

## What Is Implemented

### 1) Training Pipeline
- Script: `src/model_training.py`
- Supports configurable dataset path and target settings
- Trains multiple sklearn classifiers and selects the best by ROC-AUC
- Saves model artifact and preprocessing metadata into `models/`

### 2) Prediction Engine
- Script: `src/prediction.py`
- Handles robust preprocessing and schema alignment support
- Supports single-record and batch prediction calls
- Returns churn-focused output fields:
    - `churn_prediction`
    - `churn_probability`
    - `no_churn_probability`
    - `churn_risk_level`

### 3) Streamlit Application
- Entry point: `app/app.py`
- Pages:
    - Home
    - Predict Churn
    - Analytics
    - Model Info
    - About
- Predict Churn page supports batch CSV upload and downloadable template
- Analytics page now includes uploaded prediction analytics after scoring
- Model Info page shows real evaluation metrics and model charts

### 4) Validation and Testing
- Script: `src/test_model.py`
- Computes Accuracy, F1, ROC-AUC and pass/fail status using thresholds
- Current validated run (customer churn dataset):
    - Accuracy: `0.8228`
    - F1: `0.6232`
    - ROC-AUC: `0.8762`

### 5) Documentation and Ops
- `README.md` rewritten end-to-end
- `QUICKSTART.md` updated to match current flow
- Deployment assets available: `Dockerfile`, `docker-compose.yml`, `run.sh`

## Typical End-to-End Flow

1. Install dependencies
2. Train model (if artifacts not present)
3. Validate model quality
4. Launch Streamlit app
5. Upload CSV in Predict Churn
6. Review outputs in Predict Churn + Analytics + Model Info

## Core Commands

### Train
```bash
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
```

### Validate
```bash
python src/test_model.py --data-path data/raw/customer_data.csv --model-path models/churn_model_best.pkl --preprocessing-path models/churn_model_best_preprocessing.pkl
```

### Run app
```bash
streamlit run app/app.py
```

## Key Deliverables

- Trained churn model and preprocessing artifacts
- Churn-focused UI and outputs
- Uploaded-prediction analytics in Analytics page
- Real model evaluation visualizations in Model Info page
- Consistent, current documentation

## Next Recommended Enhancements

- Add threshold slider for churn decision boundary
- Add precision/recall metrics and PR curve in Model Info
- Add timestamp/version tags to exported predictions
- Add CI step to run `src/test_model.py` automatically

---

**Version:** 2.0  
**Last Updated:** 29 March 2026
