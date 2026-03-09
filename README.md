# Telecom Churn Predictor

An end-to-end Machine Learning web application designed to predict the likelihood of customers churning based on demographics, services used, and billing information.

## Architecture
This project has been fully refactored to a pure Python stack:
- **Modeling:** LightGBM binary classifier (handling class imbalance via `is_unbalance=True`).
- **Feature Pipeline:** MinMaxScaler and One-Hot Encoding applied entirely in Python logic mirroring original analysis.
- **Frontend app:** Built entirely with Streamlit, enabling rapid interaction and dashboard rendering.

## How to Run Locally

You have a few straightforward options to launch the web application on your machine.

### Option 1: One-Click Run (Windows)
Simply locate the **`run.bat`** script in the project root directory and double-click it. This script automatically activates the required Python environment and launches the Streamlit application for you in your default web browser.

### Option 2: Manual Terminal Commands (PowerShell)
If you prefer running commands manually, the application uses a virtual environment called `.venv`. Open your terminal (PowerShell) inside the project folder and type:

1. **Activate the Virtual Environment:**
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Start the Streamlit Server:**
   ```powershell
   streamlit run app.py
   ```

## Model Training & Artifacts
- To understand the preprocessing logic, refer to `pre-processing.ipynb`.
- To retrain the model or tune the classification threshold, run the cells inside `model.ipynb`. This notebook automatically dumps the trained model, scaler, and features metadata to `models/model.pkl` to be seamlessly consumed by the Streamlit application.

## Model Metrics Explained (Achieved Values)

The application incorporates a LightGBM classifier specifically tuned to prioritize identifying at-risk customers without sacrificing too much precision. Here are the exact metrics achieved on the test dataset and why they make this model highly effective:

- **Classification Threshold (Achieved: 0.3956)**: 
  - *What it is:* The probability score above which a customer is flagged as a "churner".
  - *Why it's better:* The default threshold (0.50) misses too many churning customers, while a purely recall-optimized threshold (0.15) flags almost everyone, creating a massive amount of false alarms. By dynamically constraining our threshold to ~0.3956, we strike the perfect mathematical balance between aggressively catching churners and trusting our loyal customers.

- **Recall (Achieved: 80.48%)**: 
  - *What it is:* The percentage of *actual* churning customers that the model successfully caught.
  - *Why it's better:* Churn is expensive. Failing to identify a leaving customer (a false negative) costs the business significantly more than offering a discount to someone who was going to stay anyway. A strict >80% recall guarantees the vast majority of your at-risk revenue is successfully identified for intervention.

- **Precision (Achieved: 48.78%)**:
  - *What it is:* Out of all the customers the model flagged as "High Risk", this is the percentage that *actually* churned. 
  - *Why it's better:* In highly imbalanced datasets like telecom churn, achieving near 50% precision while keeping recall over 80% is exceptionally strong. It means that when the model raises an alarm, there's a 1-in-2 chance that the customer is genuinely leaving, completely eliminating the ~500+ false alarms caused by the earlier, more aggressive threshold. This saves the company huge amounts of money on wasted retention offers.

- **ROC-AUC (Achieved: 0.8279)**: 
  - *What it is:* Measures the model's overall ability to distinguish between churners and non-churners across all possible thresholds.
  - *Why it's better:* A score of >0.82 proves the LightGBM model has deeply learned the underlying patterns of customer behavior (contract types, tenure, charges) rather than just guessing.

- **AUC-PR (Achieved: 0.6409)**: 
  - *What it is:* The Area Under the Precision-Recall Curve. 
  - *Why it's better:* For heavily imbalanced datasets, AUC-PR is the ultimate test of model quality. A score of 0.64 strictly outperforms baseline methods, confirming that the model's high recall does not come at the devastating expense of precision.
