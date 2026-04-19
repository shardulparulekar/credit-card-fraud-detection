#!/bin/bash
set -e

echo "=============================="
echo "  Fraud Detection — Starting"
echo "=============================="

# Run ML pipeline if model doesn't exist yet
# pipeline.py handles Kaggle download automatically using
# KAGGLE_USERNAME and KAGGLE_KEY environment variables
if [ ! -f "models/best_model.pkl" ]; then
  echo "No trained model found. Running pipeline..."
  python pipeline.py
else
  echo "Trained model found — skipping pipeline."
fi

# Launch FastAPI in background
echo "Starting API on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Launch Streamlit in foreground
echo "Starting Dashboard on port 8501..."
streamlit run dashboard/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
