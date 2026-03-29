#!/bin/bash

# Customer Churn Prediction - Startup Script
# Run this script to start the application

echo "[LAUNCH] Starting Customer Churn Prediction System..."
echo ""
echo "[CHECK] Prerequisites Check:"

DATA_PATH="${1:-data/raw/customer_data.csv}"

# Check Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo "[OK] Python 3 found: $(python3 --version)"

# Check pip
if ! command -v pip3 &> /dev/null
then
    echo "[ERROR] pip3 not found. Please install pip"
    exit 1
fi
echo "[OK] pip3 found: $(pip3 --version)"

echo ""
echo "[PACKAGES] Installing dependencies..."

# Install requirements
pip3 install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo "[OK] Dependencies installed"

echo ""
echo "[ML] Checking for trained model..."

if [ -f "models/churn_model_best.pkl" ]; then
    echo "[OK] Model found"
else
    echo "[WARNING] Model not found. Training new model..."
    echo "[DATA] Training dataset: ${DATA_PATH}"
    python3 src/model_training.py --data-path "${DATA_PATH}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model training failed"
        exit 1
    fi
fi

echo ""
echo "[DEPLOY] Starting Streamlit application..."
echo "[CHART] Open your browser and navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Start the app
python3 -m streamlit run app/app.py

