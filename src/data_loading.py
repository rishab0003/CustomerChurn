"""
Data Loading and Initial Exploration Module
Handles loading, inspection, and basic statistics of customer churn data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

def load_data(filepath):
    """
    Load customer churn dataset from CSV
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"[DATA] Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return None

# ============================================================================
# 2. DATA INSPECTION
# ============================================================================

def inspect_data(df):
    """
    Perform comprehensive data inspection
    
    Args:
        df (pd.DataFrame): Dataset to inspect
        
    Returns:
        dict: Dictionary containing inspection results
    """
    print("\n" + "="*70)
    print("[LIST] DATA INSPECTION REPORT")
    print("="*70)
    
    # Basic info
    print(f"\n[CHART] Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Data types
    print(f"\n[LIST] Data Types:")
    print(df.dtypes)
    
    # Missing values
    print(f"\n[WARNING] Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   [OK] No missing values found!")
    else:
        print(missing[missing > 0])
    
    # Duplicate rows
    print(f"\n[SEARCH] Duplicate Rows: {df.duplicated().sum()}")
    
    # First few rows
    print(f"\n[VIEW] First 5 Rows:")
    print(df.head())
    
    # Basic statistics
    print(f"\n[TREND] Numerical Columns Summary:")
    print(df.describe())
    
    # Categorical columns
    print(f"\n[LABEL] Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"   {col}: {df[col].nunique()} unique values")
    
    return {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'missing': df.isnull().sum(),
        'duplicates': df.duplicated().sum()
    }

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================

def analyze_target(df, target_col='Churn'):
    """
    Analyze target variable distribution (class imbalance)
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        
    Returns:
        pd.Series: Value counts of target variable
    """
    print("\n" + "="*70)
    print(f"[CHART] TARGET VARIABLE ANALYSIS: {target_col}")
    print("="*70)
    
    # Value counts
    churn_counts = df[target_col].value_counts()
    churn_pct = df[target_col].value_counts(normalize=True) * 100
    
    print(f"\n[CHART] Churn Distribution:")
    for label in churn_counts.index:
        print(f"   {label}: {churn_counts[label]:,} ({churn_pct[label]:.2f}%)")
    
    # Imbalance ratio
    imbalance_ratio = churn_counts.min() / churn_counts.max()
    print(f"\n[SCALE] Class Imbalance Ratio: {imbalance_ratio:.2%}")
    
    if imbalance_ratio < 0.3:
        print("   [WARNING] WARNING: Severe class imbalance detected!")
        print("   [TIP] TIP: Consider using weighted models or SMOTE")
    
    return churn_counts

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

def plot_target_distribution(df, target_col='Churn', save_path=None):
    """
    Plot target variable distribution
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Target column name
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    churn_counts = df[target_col].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for No, Red for Yes
    
    axes[0].bar(churn_counts.index, churn_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_title(f'{target_col} Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].set_xlabel(target_col, fontsize=12)
    
    # Add count labels on bars
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Percentage pie chart
    churn_pct = df[target_col].value_counts()
    explode = (0, 0.1)  # Explode the 'Yes' slice
    axes[1].pie(churn_pct.values, labels=churn_pct.index, autopct='%1.1f%%',
                colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title(f'{target_col} Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Plot saved to: {save_path}")
    
    plt.show()

def plot_data_types(df, save_path=None):
    """
    Visualize data type distribution
    
    Args:
        df (pd.DataFrame): Dataset
        save_path (str): Path to save plot
    """
    dtype_counts = df.dtypes.value_counts()
    
    plt.figure(figsize=(10, 5))
    plt.bar(dtype_counts.index.astype(str), dtype_counts.values, color='#3498db', edgecolor='black', linewidth=1.5)
    plt.title('Data Type Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Data Type', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(dtype_counts.values):
        plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Plot saved to: {save_path}")
    
    plt.show()

# ============================================================================
# 5. DATA QUALITY SUMMARY
# ============================================================================

def generate_data_quality_report(df, target_col='Churn'):
    """
    Generate comprehensive data quality report
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Target column name
        
    Returns:
        dict: Quality metrics
    """
    print("\n" + "="*70)
    print("[LIST] DATA QUALITY REPORT")
    print("="*70)
    
    quality_metrics = {
        'Total Rows': df.shape[0],
        'Total Columns': df.shape[1],
        'Missing Values %': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'Duplicate Rows': df.duplicated().sum(),
        'Numerical Columns': df.select_dtypes(include=[np.number]).shape[1],
        'Categorical Columns': df.select_dtypes(include=['object']).shape[1],
        'Target Variable': target_col,
        'Class Distribution': df[target_col].value_counts().to_dict()
    }
    
    for metric, value in quality_metrics.items():
        if metric != 'Class Distribution':
            print(f"[OK] {metric}: {value}")
    
    print(f"\n[OK] Class Distribution:")
    for cls, count in quality_metrics['Class Distribution'].items():
        print(f"    {cls}: {count}")
    
    return quality_metrics

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    df = load_data('data/raw/customer_data.csv')
    
    if df is not None:
        # Inspect data
        inspect_results = inspect_data(df)
        
        # Analyze target variable
        churn_counts = analyze_target(df, target_col='Churn')
        
        # Generate quality report
        quality_report = generate_data_quality_report(df, target_col='Churn')
        
        # Create plots directory if it doesn't exist
        Path('results/plots').mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        print("\n" + "="*70)
        print("[ART] GENERATING VISUALIZATIONS...")
        print("="*70)
        
        plot_target_distribution(df, target_col='Churn', 
                                save_path='results/plots/01_churn_distribution.png')
        plot_data_types(df, 
                       save_path='results/plots/02_data_types.png')
        
        print("\n[OK] Stage 2 Complete! Data loaded and analyzed successfully.")