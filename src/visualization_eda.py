"""
Advanced EDA Visualizations
Create multiple plots to understand feature relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
df = pd.read_csv('data/raw/customer_data.csv')

# Create plots directory
Path('results/plots').mkdir(parents=True, exist_ok=True)

print(\"[ART] Generating visualizations...\")

# ============================================================================
# 1. NUMERICAL FEATURES DISTRIBUTION
# ============================================================================

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numerical_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols[:4]):
        axes[idx].hist(df[col], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/03_numerical_distributions.png', dpi=300, bbox_inches='tight')
    print("[CHECK] Saved: 03_numerical_distributions.png")
    plt.close()

# ============================================================================
# 2. TENURE VS CHURN (KEY INSIGHT)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
df.boxplot(column='tenure', by='Churn', ax=axes[0])
axes[0].set_title('Tenure vs Churn (Box Plot)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Tenure (months)')
plt.sca(axes[0])
plt.xticks([1, 2], ['No', 'Yes'])

# Violin plot
sns.violinplot(data=df, x='Churn', y='tenure', ax=axes[1], palette=['#2ecc71', '#e74c3c'])
axes[1].set_title('Tenure Distribution by Churn', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Tenure (months)')

plt.tight_layout()
plt.savefig('results/plots/04_tenure_vs_churn.png', dpi=300, bbox_inches='tight')
print("[CHECK] Saved: 04_tenure_vs_churn.png")
plt.close()

# ============================================================================
# 3. CATEGORICAL FEATURES vs CHURN
# ============================================================================

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')

# Select a few key categorical features
key_categorical = categorical_cols[:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(key_categorical):
    # Create cross-tabulation
    ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[idx].set_title(f'{col} vs Churn (%)', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Percentage (%)')
    axes[idx].legend(title='Churn', labels=['No', 'Yes'])
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/05_categorical_vs_churn.png', dpi=300, bbox_inches='tight')
print("[CHECK] Saved: 05_categorical_vs_churn.png")
plt.close()

# ============================================================================
# 4. MONTHLY CHARGES VS CHURN
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
axes[0].scatter(df[df['Churn']=='No']['MonthlyCharges'], 
               df[df['Churn']=='No']['tenure'], 
               alpha=0.5, label='No Churn', color='#2ecc71', s=30)
axes[0].scatter(df[df['Churn']=='Yes']['MonthlyCharges'], 
               df[df['Churn']=='Yes']['tenure'], 
               alpha=0.5, label='Churn', color='#e74c3c', s=30)
axes[0].set_xlabel('Monthly Charges ($)')
axes[0].set_ylabel('Tenure (months)')
axes[0].set_title('Monthly Charges vs Tenure by Churn', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Distribution by churn
sns.kdeplot(data=df[df['Churn']=='No'], x='MonthlyCharges', ax=axes[1], 
           label='No Churn', color='#2ecc71', fill=True, alpha=0.5, linewidth=2)
sns.kdeplot(data=df[df['Churn']=='Yes'], x='MonthlyCharges', ax=axes[1], 
           label='Churn', color='#e74c3c', fill=True, alpha=0.5, linewidth=2)
axes[1].set_xlabel('Monthly Charges ($)')
axes[1].set_ylabel('Density')
axes[1].set_title('Monthly Charges Distribution by Churn', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/06_monthly_charges_analysis.png', dpi=300, bbox_inches='tight')
print("[CHECK] Saved: 06_monthly_charges_analysis.png")
plt.close()

print("\n[OK] All visualizations generated successfully!")
print("[FOLDER] Saved to: results/plots/")