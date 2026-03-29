"""
Detailed EDA Analysis - Run in Jupyter or as script
This provides deeper insights into each feature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
df = pd.read_csv('data/raw/customer_data.csv')

print("="*70)
print("EXPLORATORY DATA ANALYSIS - DETAILED")
print("="*70)

# ============================================================================
# CATEGORICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*70)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target from this analysis
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')

for col in categorical_cols[:5]:  # Show first 5 to avoid clutter
    print(f"\n[LABEL] {col}:")
    print(f"   Unique values: {df[col].nunique()}")
    print(f"   Distribution:")
    print(df[col].value_counts())

# ============================================================================
# NUMERICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("NUMERICAL FEATURES ANALYSIS")
print("="*70)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols[:5]:  # Show first 5
    print(f"\n[CHART] {col}:")
    print(f"   Mean: {df[col].mean():.2f}")
    print(f"   Median: {df[col].median():.2f}")
    print(f"   Std Dev: {df[col].std():.2f}")
    print(f"   Min: {df[col].min():.2f}")
    print(f"   Max: {df[col].max():.2f}")

# ============================================================================
# CHURN RELATIONSHIP ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("CHURN RELATIONSHIP WITH FEATURES")
print("="*70)

# Numerical features vs Churn
print("\n[TREND] Numerical Features vs Churn:")
for col in numerical_cols[:3]:
    churn_yes = df[df['Churn'] == 'Yes'][col].mean()
    churn_no = df[df['Churn'] == 'No'][col].mean()
    diff = ((churn_yes - churn_no) / churn_no) * 100
    print(f"   {col}:")
    print(f"      Churn=Yes avg: {churn_yes:.2f}")
    print(f"      Churn=No avg: {churn_no:.2f}")
    print(f"      Difference: {diff:.2f}%")

print("\n[OK] EDA Analysis Complete!")