# Banking Fraud Detection System

## Executive Summary

This project implements a machine learning-based fraud detection system for banking transactions. The goal is to identify fraudulent transactions with high accuracy, specifically focusing on minimizing false negatives (missed fraud cases).

### Exploratory Data Analysis (EDA) Insights
- **Imbalanced Dataset:** The dataset is highly imbalanced, with fraudulent transactions representing a very small fraction of the total.
- **Transaction Types:** Fraudulent transactions occur primarily in `TRANSFER` and `CASH_OUT` types.
- **Feature Importance:** Key features include transaction amount and balance changes in origin and destination accounts.

### Model Selection
I evaluated multiple models including Logistic Regression, Decision Trees, Random Forest, and Isolation Forest.
- **Selected Model:** **Balanced Random Forest Classifier**
- **Reason for Choice:** It achieved the highest **Recall (~99.7%)**, which is the most critical metric for fraud detection. High recall ensures that we catch almost all fraudulent transactions, even if it means a slightly higher false positive rate (which can be reviewed manually).

## Project Structure

```
banking-fraud-detection-ml/
├─ app.py                 <-- Streamlit web application
├─ requirements.txt       <-- Python dependencies
├─ model/                 <-- Model artifacts
│   └─ best_model_Recall.pkl
├─ .gitignore             <-- Git ignore file
└─ README.md              <-- Project documentation
```

## Setup Instructions

1.  **Clone the repository** 
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
2.  **Interact with the Interface**:
    - Enter transaction details in the sidebar.
    - Click "Predict Fraud Status".
    - View the prediction result and transaction summary.

## Model Details
- **Input Features:** `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`.
- **Preprocessing:** Categorical `type` is encoded.
