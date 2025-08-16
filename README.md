# Fraud Detection with AutoEncoder (PyOD 2.0.5)

This project implements a **fraud detection system using Autoencoders from the PyOD library** on the Kaggle **Credit Card Fraud Detection dataset**.  
The model learns normal transaction patterns and detects anomalies (potential fraud) via reconstruction error.

## 📖 Description
- **Dataset**: Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`)  
- **Library**: [PyOD 2.0.5](https://github.com/yzhao062/pyod)  
- **Model**: Deep AutoEncoder trained with contamination prior  
- **Outputs**: Confusion matrix, anomaly score histogram, and classification metrics  

---

## ⚙️ Technical Requirements
- Python 3.9+  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

Dependencies include:
- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- pyod (2.0.5)  
- tensorflow / keras  
- torch  
- tqdm  
- torchvision
- torchaudio  

---

## ▶️ How to Run the Application

1. **Clone repository**:
   ```bash
   git clone https://github.com/Hanuman42109/fraud_auto_encoder_pyod.git
   cd fraud_auto_encoder_pyod
   ```

2. **Download dataset** (not included in repo because it’s >100 MB):  
   - Get `creditcard.csv` from Kaggle  
   - Place it under `./data/creditcard.csv`

3. **Run script**:
   ```bash
   python src/fraud_autoencoder_pyod.py --data ./data/creditcard.csv --outputs ./outputs
   ```

4. **View results** in `./outputs/`:
   - `metrics.txt` → Precision, Recall, F1, ROC-AUC  
   - `confusion_matrix.png` → Predicted vs actual  
   - `score_histogram.png` → Score distribution (Legit vs Fraud)  

---

## 📌 Notes
- `data/creditcard.csv` is excluded via `.gitignore` (too large for GitHub).  
- The default contamination rate is `0.002`. Adjust if you use different splits.  
- Training config (hidden layers, epochs, batch size) can be modified in the script for experimentation.  

---

## 🔗 Repository
[https://github.com/Hanuman42109/fraud_auto_encoder_pyod](https://github.com/Hanuman42109/fraud_auto_encoder_pyod)
