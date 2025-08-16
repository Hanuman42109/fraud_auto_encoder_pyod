#!/usr/bin/env python3
"""
Fraud Detection with AutoEncoder (PyOD 2.0.5)
----------------------------------------------
- Dataset: Kaggle "Credit Card Fraud Detection" (creditcard.csv)
- Model: PyOD AutoEncoder (v2.0.5)
- Author: Hanuman Sai Chanukya Srinivas Chilamkuri
- Date: 2025-08-15

Usage:
    python src/fraud_autoencoder_pyod.py --data ./data/creditcard.csv --outputs ./outputs
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from pyod.models.auto_encoder import AutoEncoder

def load_data(csv_path: str):
    """Load dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Place creditcard.csv under ./data")
    df = pd.read_csv(csv_path)
    if 'Class' not in df.columns:
        raise ValueError("Expected 'Class' column indicating fraud labels (0=legit, 1=fraud).")
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int).values
    return X, y

def scale_features(X: pd.DataFrame):
    """Standardize features to zero mean, unit variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler

def train_autoencoder(X_train: np.ndarray, contamination: float = 0.002, random_state: int = 42):
    """
    Train AutoEncoder using PyOD 2.0.5 API.
    In this version, use `train_epochs` and `train_batch_size` in __init__().
    """
    clf = AutoEncoder(
        hidden_neuron_list=[64, 32, 32, 64],
        contamination=contamination,
        verbose=1,
        random_state=random_state,
        epoch_num=50,       # number of epochs
        batch_size=256,
        dropout_rate=0.2
    )
    clf.fit(X_train)
    return clf

def evaluate_and_plot(clf, X_test, y_test, out_dir: str):
    """Evaluate model and save confusion matrix + score histogram."""
    os.makedirs(out_dir, exist_ok=True)
    y_pred = clf.predict(X_test)  # 0=inlier, 1=outlier
    scores = clf.decision_function(X_test)

    # Classification metrics
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, scores)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("Classification Report\n")
        f.write(report + "\n")
        f.write(f"ROC-AUC: {auc:.6f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit (0)", "Fraud (1)"])
    fig1, ax1 = plt.subplots()
    disp.plot(ax=ax1, values_format='d', colorbar=False)
    ax1.set_title("AutoEncoder (PyOD 2.0.5) — Confusion Matrix")
    fig1.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig1.savefig(cm_path, dpi=160)
    plt.close(fig1)

    # Score histogram
    fig2, ax2 = plt.subplots()
    ax2.hist(scores[y_test==0], bins=50, alpha=0.7, label="Legit")
    ax2.hist(scores[y_test==1], bins=50, alpha=0.7, label="Fraud")
    ax2.set_title("Anomaly Score Distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    fig2.tight_layout()
    hist_path = os.path.join(out_dir, "score_histogram.png")
    fig2.savefig(hist_path, dpi=160)
    plt.close(fig2)

    return report, auc, cm_path, hist_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/creditcard.csv")
    parser.add_argument("--outputs", type=str, default="./outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.002)
    args = parser.parse_args()

    print("[1/5] Loading data...")
    X_df, y = load_data(args.data)

    print("[2/5] Scaling features...")
    X_scaled, scaler = scale_features(X_df)

    print("[3/5] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"[Info] Train size: {X_train.shape}, Test size: {X_test.shape}")

    print("[4/5] Training AutoEncoder...")
    clf = train_autoencoder(X_train, contamination=args.contamination, random_state=args.random_state)

    print("[5/5] Evaluating...")
    report, auc, cm_path, hist_path = evaluate_and_plot(clf, X_test, y_test, args.outputs)

    print("\n=== RESULTS ===")
    print(report)
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved histogram to: {hist_path}")

if __name__ == "__main__":
    main()
