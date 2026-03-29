# Deployment Guide (Customer Churn Prediction)

This guide explains how to deploy the churn prediction app locally, with Docker, and on cloud platforms.

## 1) Local Deployment

### Option A: Recommended manual setup

```bash
cd customer_churn_prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
python src/test_model.py --data-path data/raw/customer_data.csv --model-path models/churn_model_best.pkl --preprocessing-path models/churn_model_best_preprocessing.pkl
streamlit run app/app.py
```

Open: `http://localhost:8501`

### Option B: helper script

```bash
chmod +x run.sh
./run.sh
```

## 2) Docker Deployment

### Build and run (single container)

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

### Docker Compose

```bash
docker-compose up --build
```

Background mode:

```bash
docker-compose up -d --build
```

Stop:

```bash
docker-compose down
```

## 3) Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app using:
   - **Repository**: your repo
   - **Branch**: `main`
   - **Main file path**: `app/app.py`
4. Deploy.

### Notes
- Ensure model files are available during runtime (`models/churn_model_best.pkl` and preprocessing file).
- If model files are not committed, add a startup step to train/download them.

## 4) VM/Server Deployment (Linux)

Use this for EC2, GCE, Azure VM, or any Linux host.

```bash
git clone <your-repo-url>
cd customer_churn_prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

Expose port `8501` in firewall/security group.

## 5) Production Hardening

### Run behind reverse proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Keep process alive (systemd example)

Create `/etc/systemd/system/churn-app.service`:

```ini
[Unit]
Description=Customer Churn Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/customer_churn_prediction
ExecStart=/home/ubuntu/customer_churn_prediction/venv/bin/streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable churn-app
sudo systemctl start churn-app
sudo systemctl status churn-app
```

## 6) Deployment Verification Checklist

- App opens successfully in browser
- Model artifacts exist:
  - `models/churn_model_best.pkl`
  - `models/churn_model_best_preprocessing.pkl`
- Predict Churn page runs batch predictions
- Analytics page shows uploaded prediction analytics
- Model Info page shows metrics + confusion matrix + ROC

## 7) Troubleshooting

### Port already in use

```bash
lsof -i :8501
kill -9 <PID>
```

Or use another port:

```bash
streamlit run app/app.py --server.port 8502
```

### Missing dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Model not found

```bash
python src/model_training.py --data-path data/raw/customer_data.csv --target-col Churn --positive-label Yes
```

### Container logs

```bash
docker-compose logs -f
```

## 8) Maintenance

- Retrain periodically with fresh data.
- Re-run validation script after retraining.
- Track key metrics (Accuracy/F1/ROC-AUC) over time.
- Keep dependencies updated.

For complete usage details, see [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).
