# Taxi Service Acceptance Prediction

Streamlit app that predicts whether a taxi driver will accept a trip request and provides model explanations (SHAP/LIME). Optional LLM recommendations are generated using a Groq-hosted OpenAI-compatible API.

## Project structure

- frontend/app.py — Streamlit UI and inference logic
- models/ — Trained model files
- requirements.txt — Python dependencies

## Features

- Predict accept/reject with probability
- Model selection: XGBoost, LightGBM, CatBoost
- Local SHAP, Local LIME, Global SHAP analytics (when installed)
- Optional LLM recommendations via Groq API
- Demo mode when model files are missing

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   
   pip install -r requirements.txt

3. (Optional) Enable LLM recommendations:
   
   Set an environment variable named GROQ_API_KEY with your Groq API key.

## Run the app

From the project root:

streamlit run frontend/app.py

## Models

Expected model files (relative to project root):

- models/xgboost_model.pkl
- models/lightgbm_model.pkl
- models/catboost_model.pkl

If a model file is missing, the app falls back to demo mode with randomized predictions.

## Notes

- If you see a model loading error, it is likely a scikit-learn version mismatch. Use the versions in requirements.txt.
- SHAP and LIME plots require shap, lime, and matplotlib to be installed (already listed in requirements.txt).

## Disclaimer

This tool is for educational and analytical purposes only. Predictions are probabilistic and should be used alongside operational judgment.
