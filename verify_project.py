#!/usr/bin/env python3
"""
Project Verification Script
Checks if all required files and dependencies are in place
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "[OK]" if exists else "[ERROR]"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory(dirpath, description):
    """Check if a directory exists"""
    exists = Path(dirpath).exists()
    status = "[OK]" if exists else "[WARNING]"
    print(f"{status} {description}: {dirpath}/")
    return exists

def check_imports():
    """Check if required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'streamlit',
        'matplotlib',
        'seaborn',
        'plotly',
        'joblib'
    ]
    
    print("\n[PACKAGES] Checking Python Packages:")
    print("=" * 60)
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[ERROR] {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def main():
    """Main verification"""
    print("\n" + "=" * 60)
    print("[CHECK] PROJECT VERIFICATION")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check required files
    print("\n[FILES] Checking Required Files:")
    print("=" * 60)
    
    required_files = {
        'requirements.txt': 'Requirements file',
        'README.md': 'README documentation',
        'QUICKSTART.md': 'Quick start guide',
        'DEPLOYMENT.md': 'Deployment guide',
        'CONFIG_EXAMPLES.py': 'Configuration examples',
        'setup_project.py': 'Setup script',
        'app/app.py': 'Streamlit application',
        'src/data_loading.py': 'Data loading module',
        'src/prediction.py': 'Prediction module',
        'src/model_training.py': 'Model training script',
        'src/visualization_eda.py': 'Visualization module',
        '.streamlit/config.toml': 'Streamlit configuration',
        'Dockerfile': 'Docker configuration',
        'docker-compose.yml': 'Docker Compose file',
    }
    
    files_ok = all(check_file(f, desc) for f, desc in required_files.items())
    
    # Check required directories
    print("\n[FOLDERS] Checking Required Directories:")
    print("=" * 60)
    
    required_dirs = {
        'app': 'App directory',
        'src': 'Source code directory',
        'data': 'Data directory',
        'data/raw': 'Raw data directory',
        'data/processed': 'Processed data directory',
        'models': 'Models directory',
        'notebooks': 'Notebooks directory',
        'results': 'Results directory',
        '.streamlit': 'Streamlit config directory',
    }
    
    dirs_ok = all(check_directory(d, desc) for d, desc in required_dirs.items())
    
    # Check data file
    print("\n[DATA] Checking Data Files:")
    print("=" * 60)
    
    data_file = 'data/raw/customer_data.csv'
    data_exists = check_file(data_file, 'Customer dataset')
    
    if data_exists:
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"   [OK] Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   [OK] Columns: {', '.join(df.columns[:5])}...")
        except Exception as e:
            print(f"   [WARNING] Error reading data: {e}")
            data_exists = False
    
    # Check model files
    print("\n[ML] Checking Model Files:")
    print("=" * 60)
    
    model_files = {
        'models/churn_model_best.pkl': 'Trained churn model',
        'models/churn_model_best_preprocessing.pkl': 'Preprocessing pipeline'
    }
    
    models_ok = all(check_file(f, desc) for f, desc in model_files.items())
    
    # Check Python packages
    packages_ok = check_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY] VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_ok = files_ok and dirs_ok and data_exists and models_ok and packages_ok
    
    print(f"\n[OK] Required Files: {'PASS' if files_ok else 'FAIL'}")
    print(f"[OK] Required Directories: {'PASS' if dirs_ok else 'FAIL'}")
    print(f"[OK] Data Files: {'PASS' if data_exists else 'FAIL'}")
    print(f"[OK] Model Files: {'PASS' if models_ok else 'FAIL'}")
    print(f"[OK] Python Packages: {'PASS' if packages_ok else 'FAIL'}")
    
    print(f"\n{'[SUCCESS] Project Ready!' if all_ok else '[WARNING] Issues Found'}\n")
    
    # Next steps
    print("\n" + "=" * 60)
    print("[LAUNCH] NEXT STEPS")
    print("=" * 60)
    
    if all_ok:
        print("\n[OK] All checks passed! You can now:")
        print("   1. Run: ./run.sh")
        print("   2. Or: streamlit run app/app.py")
        print("   3. Open: http://localhost:8501")
    else:
        print("\n[ERROR] Some issues found. Please fix them before running:")
        if not files_ok:
            print("   - Check that all required files exist")
        if not dirs_ok:
            print("   - Create missing directories:")
            print("     python setup_project.py")
        if not data_exists:
            print("   - Ensure customer_data.csv is in data/raw/")
        if not models_ok:
            print("   - Train the model:")
            print("     python src/model_training.py")
        if not packages_ok:
            print("   - Install dependencies:")
            print("     pip install -r requirements.txt")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
