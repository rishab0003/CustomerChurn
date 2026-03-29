"""
Configuration Examples and Best Practices
Customize the application behavior with these configuration options
"""

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================
# File: .streamlit/config.toml

"""
[theme]
primaryColor = "#3498db"          # Main color (default: #FF4B4B)
backgroundColor = "#ffffff"       # Background (default: #FFFFFF)
secondaryBackgroundColor = "#f0f2f6"  # Secondary background
textColor = "#262730"            # Text color
font = "sans serif"              # Font family

[client]
toolbarMode = "viewer"           # "viewer", "minimal", or "developer"
showErrorDetails = true          # Show detailed error messages

[logger]
level = "info"                   # "debug", "info", "warning", "error", "critical"

[server]
port = 8501                      # Port number
headless = false                 # True for production
enableXsrfProtection = true      # CSRF protection
enableCORS = false               # Enable CORS
maxUploadSize = 200              # Max upload size in MB
"""

# ============================================================================
# PREDICTION CONFIGURATIONS
# ============================================================================

# Risk Level Thresholds (in prediction.py)
RISK_THRESHOLDS = {
    'low_risk': (0.0, 0.25),      # Low Risk
    'medium_risk': (0.25, 0.5),   # Medium Risk
    'high_risk': (0.5, 0.75),     # High Risk
    'very_high_risk': (0.75, 1.0) # Very High Risk
}

# Model Paths
MODEL_CONFIG = {
    'model_path': 'models/churn_model_best.pkl',
    'preprocessing_path': 'models/churn_model_best_preprocessing.pkl',
    'backup_models': [
        'models/logistic_regression.pkl',
        'models/random_forest.pkl'
    ]
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Data paths
DATA_CONFIG = {
    'raw_data': 'data/raw/customer_data.csv',
    'processed_data': 'data/processed/',
    'results_dir': 'results/plots/',
}

# Feature names and types
FEATURE_CONFIG = {
    'numerical_features': [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ],
    'categorical_features': [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ],
    'target': 'Churn',
    'id_column': 'customerID'
}

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

# Train-test split
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify_target': True,
}

# Model hyperparameters
MODEL_HYPERPARAMS = {
    'LogisticRegression': {
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs'
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
}

# Scaling configuration
SCALING_CONFIG = {
    'method': 'StandardScaler',  # or 'MinMaxScaler', 'RobustScaler'
    'feature_range': (0, 1),     # for MinMaxScaler
}

# ============================================================================
# UI/UX CONFIGURATION
# ============================================================================

# Page configuration
PAGE_CONFIG = {
    'page_title': 'Customer Churn Prediction',
    'page_icon': '[CHART]',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color scheme
COLOR_SCHEME = {
    'no_churn': '#2ecc71',        # Green
    'churn': '#e74c3c',           # Red
    'neutral': '#3498db',         # Blue
    'warning': '#f39c12',         # Orange
    'danger': '#c0392b'           # Dark Red
}

# ============================================================================
# PRODUCTION CONFIGURATION
# ============================================================================

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/app.log',
    'max_size': '10MB',
    'backup_count': 5
}

# Database configuration (if using)
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'churn_prediction',
    'user': 'churn_user',
    'password': 'secure_password',  # Use environment variables!
    'table': 'predictions'
}

# ============================================================================
# BATCH PREDICTION CONFIGURATION
# ============================================================================

BATCH_CONFIG = {
    'max_records': 10000,          # Max records per batch
    'chunk_size': 1000,            # Process in chunks
    'timeout': 300,                # Timeout in seconds
    'save_results': True,          # Save results to file
    'output_format': 'csv'         # csv, json, parquet
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Feature transformations
FEATURE_ENGINEERING = {
    'create_tenure_bins': True,
    'tenure_bins': [0, 6, 12, 24, float('inf')],
    'create_charge_bins': True,
    'charge_bins': [0, 50, 100, 150, float('inf')],
    'create_interactions': False,
    'interaction_pairs': [
        ('Contract', 'MonthlyCharges'),
        ('tenure', 'InternetService')
    ]
}

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

IMBALANCE_CONFIG = {
    'handle_imbalance': True,
    'method': 'class_weights',  # 'class_weights', 'smote', 'oversample'
    'positive_weight': 3.0,     # Weight for minority class
}

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

OPTIMIZATION_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,           # Cache time-to-live in seconds
    'enable_lazy_loading': True,
    'predict_batch_size': 100,   # Batch size for predictions
    'enable_gpu': False,          # GPU acceleration (if available)
}

# ============================================================================
# API CONFIGURATION (Future Enhancement)
# ============================================================================

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 4,
    'timeout': 60,
    'max_connections': 100,
    'rate_limit': '100/minute'   # Requests per minute
}

# ============================================================================
# EXAMPLE: How to Use Configurations
# ============================================================================

"""
# In your Python code:

from config import RISK_THRESHOLDS, MODEL_CONFIG, COLOR_SCHEME

# Use risk thresholds
if churn_probability > RISK_THRESHOLDS['very_high_risk'][0]:
    risk_level = 'Very High Risk'

# Use model paths
model_path = MODEL_CONFIG['model_path']

# Use colors
st.write(f"<span style='color:{COLOR_SCHEME['churn']}'>High Churn Risk</span>", 
         unsafe_allow_html=True)

# Use feature config
numerical_cols = FEATURE_CONFIG['numerical_features']

# Use model hyperparameters
from sklearn.ensemble import RandomForest
rf_params = MODEL_HYPERPARAMS['RandomForest']
model = RandomForest(**rf_params)
"""

# ============================================================================
# ENVIRONMENT VARIABLES (Best Practice)
# ============================================================================

"""
# Create a .env file:

# Model paths
MODEL_PATH=models/churn_model_best.pkl
PREPROCESSING_PATH=models/churn_model_best_preprocessing.pkl

# Data paths
DATA_PATH=data/raw/customer_data.csv
RESULTS_PATH=results/plots/

# Server config
SERVER_PORT=8501
SERVER_ADDRESS=0.0.0.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Database (if using)
DATABASE_URL=postgresql://user:password@localhost:5432/churn_db

# API Keys
API_KEY=your_api_key_here

# Environment
ENVIRONMENT=production  # or development, staging
"""

# ============================================================================
# SECRETS MANAGEMENT (Streamlit)
# ============================================================================

"""
# Create .streamlit/secrets.toml:

# Database credentials
[database]
host = "prod-db.example.com"
port = 5432
dbname = "churn_prediction"
user = "db_user"
password = "secure_password"

# API Keys
[api]
key = "your_api_key_here"
secret = "your_api_secret"

# Email config
[email]
sender = "noreply@example.com"
smtp_server = "smtp.gmail.com"
smtp_port = 587
password = "your_email_password"

# Access in code:
import streamlit as st
db_config = st.secrets["database"]
api_key = st.secrets["api"]["key"]
"""
