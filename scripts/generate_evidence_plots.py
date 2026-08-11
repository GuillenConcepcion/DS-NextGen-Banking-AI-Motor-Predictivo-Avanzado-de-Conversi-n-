import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import mlflow
import joblib
import os
import sys

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")

def load_data():
    df = pd.read_csv("data/raw/bank_marketing.csv")
    X = df.drop('y', axis=1)
    y = df['y'].replace({'no': 0, 'yes': 1})
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    v2_features = ['default', 'housing', 'loan', 'day', 'contact', 'month', 'campaign', 'pdays']
    return X_test[v2_features], y_test

def main():
    print("Loading model and data...")
    try:
        model = mlflow.pyfunc.load_model("models:/Catboost_Simpler_Cols@champion")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    X_test, y_test = load_data()
    
    # 1. Prediction Probabilities
    print("Generating predictions...")
    y_probs = model.predict(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 2. Confusion Matrix
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Deposit', 'Deposit'], 
                yticklabels=['No Deposit', 'Deposit'])
    plt.title('Confusion Matrix - Champion Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 3. ROC Curve
    print("Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png', dpi=300)
    plt.close()
    
    # 4. SHAP Summary Plot
    # SHAP is currently throwing numba compatibility issues in this isolated script,
    # but the Streamlit App handles it well. We skip SHAP summary plot here.
    
    print("All evidence graphics generated and saved in 'reports/figures/'.")

if __name__ == "__main__":
    main()
