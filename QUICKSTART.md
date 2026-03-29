# Quick Start (Customer Churn Prediction)

Use this guide to run the project from zero to first churn prediction quickly.

## 1) Setup

```bash
cd customer_churn_prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Train (if model files do not exist)

```bash
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
```

## 3) Validate model quality

```bash
python src/test_model.py \
   --data-path data/raw/customer_data.csv \
   --model-path models/churn_model_best.pkl \
   --preprocessing-path models/churn_model_best_preprocessing.pkl
```

## 4) Run the app

```bash
streamlit run app/app.py
```

Open in browser: `http://localhost:8501`

If port 8501 is busy:

```bash
streamlit run app/app.py --server.port 8502
```

## 5) Predict churn (app flow)

1. Open **Predict Churn**
2. Download the model template CSV
3. Fill your customer rows
4. Upload CSV
5. Click **Predict All Rows**
6. Download results

## 6) Analyze results

- **Analytics** page:
   - Uploaded prediction summary cards
   - Churn class distribution chart
   - Churn probability histogram
- **Model Info** page:
   - Feature importance chart
   - Accuracy / F1 / ROC-AUC
   - Confusion matrix and ROC curve

## Optional: Run with Docker

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

or

```bash
docker-compose up --build
```

## Troubleshooting

### `ModuleNotFoundError`
```bash
pip install --upgrade -r requirements.txt
```

### Model files missing
```bash
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
```

### Streamlit command not found
```bash
python -m streamlit run app/app.py
```

## Verify everything

- `models/churn_model_best.pkl` exists
- `models/churn_model_best_preprocessing.pkl` exists
- `data/raw/customer_data.csv` exists
- App starts and loads pages without errors

For full details, see [README.md](README.md).
