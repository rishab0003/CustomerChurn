from pathlib import Path

folders = [
    'data/raw',
    'data/processed',
    'notebooks',
    'src',
    'models',
    'app',
    'results/plots'
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created: {folder}/")

print("\n[OK] Project structure created successfully!")
print("\nYour project now looks like:")
print("""
customer_churn_prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── app/
├── results/
│   └── plots/
├── venv/
├── requirements.txt
└── setup_project.py
""")