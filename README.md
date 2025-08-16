# Fraud Detection with AutoEncoder (PyOD)

This project trains a PyOD AutoEncoder on the Kaggle **Credit Card Fraud Detection** dataset to identify anomalous transactions via reconstruction error.

## Quickstart (VS Code)

1. **Clone or download this folder** into your machine.
2. Open in **VS Code**.
3. Create a virtual environment and install deps:
   ```bash
   python -m venv .env
   # Windows PowerShell
   .env\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Add data**: download `creditcard.csv` from Kaggle and place it under `./data`.
5. **Run**:
   ```bash
   python src/fraud_autoencoder_pyod.py --data ./data/creditcard.csv --outputs ./outputs
   ```

Artifacts are saved in `./outputs`:
- `metrics.txt`
- `confusion_matrix.png`
- `score_histogram.png`
- `run_manifest.json`

## Notes
- The script uses `contamination=0.002` as a prior. Adjust if using a different dataset split.
- Tune `hidden_neurons`, `epochs`, and `batch_size` for better performance.
