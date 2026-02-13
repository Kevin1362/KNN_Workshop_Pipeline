# K-Nearest Neighbors (KNN) Workshop — Pipeline Pattern (Student Version)

This repo demonstrates **KNN classification** implemented **from scratch** and structured using a simple **Machine Learning Pipeline Pattern**.

## What’s inside
- `notebooks/knn_workshop.ipynb` — presentation-friendly notebook (CSV + evaluation + plots)
- `src/` — reusable pipeline components (data loading, preprocessing, model, evaluation)
- `scripts/` — command-line runners

## Quick start (Windows / Mac / Linux)

### 1) Create & activate a virtual environment
**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Optional) Create a Jupyter kernel named `knn`
```bash
python -m ipykernel install --user --name knn --display-name "knn"
```

### 4) Run the notebook
```bash
jupyter notebook
```
Open: `notebooks/knn_workshop.ipynb`

### 5) Run from the terminal (no notebook)
```bash
python scripts/run_csv_iris.py
```

## Notes
- The notebook uses the **Iris** dataset (bundled via scikit-learn).
- API + database loaders are included for learning (optional).
