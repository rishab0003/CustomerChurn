"""
Test script to verify all installations and setup are correct
"""

import sys

print("=" * 60)
print("TESTING PROJECT SETUP")
print("=" * 60)

# Test 1: Check Python version
print("\n1. Python Version:")
print(f"   [OK] {sys.version}")

# Test 2: Check required packages
print("\n2. Required Packages:")
packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'matplotlib', 'seaborn', 'joblib', 'streamlit']

for package in packages:
    try:
        __import__(package)
        print(f"   [OK] {package} installed")
    except ImportError:
        print(f"   [ERROR] {package} NOT installed")

# Test 3: Check folder structure
from pathlib import Path
print("\n3. Project Folders:")
folders = ['data/raw', 'data/processed', 'notebooks', 'src', 'models', 'app', 'results']

for folder in folders:
    if Path(folder).exists():
        print(f"   [OK] {folder}/ exists")
    else:
        print(f"   [ERROR] {folder}/ missing")

# Test 4: Check if dataset exists
print("\n4. Dataset:")
data_path = Path('data/raw/customer_data.csv')
if data_path.exists():
    print(f"   [OK] Dataset found: {data_path}")
else:
    print(f"   [ERROR] Dataset NOT found - download or create it!")

print("\n" + "=" * 60)
print("SETUP TEST COMPLETE")
print("=" * 60)