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

## Results

Confution Metrix

Random Forest 
<img width="901" height="721" alt="image" src="https://github.com/user-attachments/assets/25f2095d-3059-4b5e-a4be-1d6aa9ddcae6" />
Random Forest Accuracy: 0.8335
Random Forest Precision: 0.8326
Random Forest Recall: 0.7336
Random Forest F1-Score: 0.7800
Random Forest ROC AUC Score: 0.9160

XGBoost
<img width="940" height="753" alt="image" src="https://github.com/user-attachments/assets/ac729033-a7cb-470b-8338-abbe94d1d5c9" />
XGBoost Accuracy: 0.8346
XGBoost Precision: 0.8315
XGBoost Recall: 0.7386
XGBoost F1-Score: 0.7823
XGBoost ROC AUC Score: 0.9201

LiteGBM
<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/23728e02-0789-44f3-abd6-3ea98a1f29c5" />
LiteGBM Accuracy: 0.8344
LiteGBM Precision: 0.8296
LiteGBM Recall: 0.7407
LiteGBM F1-Score: 0.7827
LiteGBM ROC AUC Score: 0.9191

CatBoost
<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/bd4e0460-b7e1-4255-9742-9b81da62a7f1" />
CatBoost Accuracy: 0.8352
CatBoost Precision: 0.8353
CatBoost Recall: 0.7354
CatBoost F1-Score: 0.7822
CatBoost ROC AUC Score: 0.9201

ROC AUC Curve
<img width="779" height="790" alt="image" src="https://github.com/user-attachments/assets/1808711b-4e67-4d2f-b480-186efa3d1308" />

---
- Local SHAP
<img width="510" height="275" alt="image" src="https://github.com/user-attachments/assets/312d7970-0016-4b6c-808e-f823522b9c19" />

- Globle SHAP
<img width="604" height="424" alt="image" src="https://github.com/user-attachments/assets/8fa47b98-94a1-44d9-949a-b3c5ab012b5a" />

- LIME
<img width="506" height="275" alt="image" src="https://github.com/user-attachments/assets/e376a0e1-8516-4f7a-baeb-e7435f2805da" />



## Disclaimer

This tool is for educational and analytical purposes only. Predictions are probabilistic and should be used alongside operational judgment.
