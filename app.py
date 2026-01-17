import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Taxi Service Prediction",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .main {
        padding: 2rem;
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 45%, #f8fafc 100%);
    }
    .stSidebar {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .stSidebar * {
        color: #e2e8f0 !important;
    }
    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.2);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1E88E5 0%, #2563eb 100%);
        color: white;
        font-weight: bold;
        padding: 0.6rem;
        border-radius: 6px;
        border: 0;
        box-shadow: 0 6px 16px rgba(30, 136, 229, 0.25);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    .accept {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
    }
    .reject {
        background-color: #ffebee;
        border: 2px solid #e53935;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(6px);
        margin-bottom: 1rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        text-align: center;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }
    .metric-card span {
        display: block;
        margin-top: 0.25rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
    @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 600px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model
FEATURES = [
    'trip_fare_in_lkr', 'is_cash_trip', 'is_batch', 'est_speed',
    'fare_per_min_calc', 'driver_historical_acc_rate', 'geo_encoded', 'dist_ordinal'
]

MODEL_FILES = {
    #'random_forest': 'random_forest_model.pkl',
    'xgboost': 'models\\xgboost_model.pkl',
    'lightgbm': 'models\\lightgbm_model.pkl',
    'catboost': 'models\\catboost_model.pkl'
}

@st.cache_resource
def load_model(model_key):
    """Load the selected trained model"""
    try:
        model_filename = MODEL_FILES[model_key]
        model_path = Path(__file__).parent.parent / model_filename
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except FileNotFoundError:
        st.warning(f"⚠️ {MODEL_FILES[model_key]} not found. Using demo mode.")
        return None
    except Exception as exc:
        st.error(
            "⚠️ Model load failed. This is usually caused by a scikit-learn version mismatch. "
            "Update your environment to match the version used to train the model."
        )
        st.caption(f"Details: {exc}")
        return None

def preprocess_input(data):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([data])
    return df[FEATURES]

def _pipeline_fallback_predict(model, processed_data):
    """Fallback prediction path for sklearn Pipeline compatibility issues"""
    if not hasattr(model, "steps"):
        return None
    try:
        preprocessor = model[:-1]
        estimator = model.steps[-1][1]
        transformed = preprocessor.transform(processed_data)
        preds = estimator.predict(transformed)
        probas = None
        if hasattr(estimator, "predict_proba"):
            probas = estimator.predict_proba(transformed)[:, 1]
        return preds, probas
    except Exception:
        return None

def predict_acceptance(input_data, model):
    """Make acceptance prediction"""
    if model is None:
        # Demo mode - random prediction
        return np.random.choice([0, 1], p=[0.4, 0.6]), np.random.random()
    
    processed_data = preprocess_input(input_data)
    try:
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed_data)[0][1]
        else:
            probability = prediction
        
        return prediction, probability
    except AttributeError as exc:
        if "__sklearn_tags__" in str(exc):
            fallback = _pipeline_fallback_predict(model, processed_data)
            if fallback is not None:
                preds, probas = fallback
                prediction = preds[0]
                probability = probas[0] if probas is not None else prediction
                return prediction, probability
        st.error("⚠️ Prediction failed due to model compatibility. Please use a different model or retrain with current versions.")
        st.caption(f"Details: {exc}")
        return np.random.choice([0, 1], p=[0.4, 0.6]), np.random.random()

def generate_recommendation(input_data, signals):
    """Generate a recommendation using Groq-hosted OpenAI-compatible API"""
    if OpenAI is None:
        return "LLM client not available. Please install the openai package."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set. Add it to your environment to enable recommendations."

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    signal_text = ", ".join(signals) if signals else "no major negative signals detected"
    prompt = (
        "You are assisting a taxi dispatch team. The model predicts a driver may reject a trip. "
        "Provide 3 short, practical recommendations to increase acceptance. "
        "Do a short justification of the recommendations based on the trip details and signals provided. "
        f"Trip details: {input_data}. Signals: {signal_text}."
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def generate_demo_input():
    """Generate randomized demo input values"""
    return {
        'trip_fare_in_lkr': float(np.round(np.random.uniform(150, 3500), 2)),
        'is_cash_trip': int(np.random.choice([0, 1])),
        'is_batch': int(np.random.choice([0, 1])),
        'est_speed': float(np.round(np.random.uniform(5, 70), 1)),
        'fare_per_min_calc': float(np.round(np.random.uniform(5, 80), 1)),
        'driver_historical_acc_rate': float(np.round(np.random.uniform(0.1, 0.98), 2)),
        'geo_encoded': int(np.random.randint(0, 51)),
        'dist_ordinal': int(np.random.randint(0, 7))
    }

def apply_demo_values():
    """Apply randomized demo values to session state before widgets render"""
    demo_values = generate_demo_input()
    for key, value in demo_values.items():
        st.session_state[key] = value
    st.session_state["run_prediction"] = True

def _get_estimator(model):
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model

def _get_preprocessor(model):
    if hasattr(model, "steps"):
        return model[:-1]
    return None

def _transform_if_needed(model, df):
    preprocessor = _get_preprocessor(model)
    if preprocessor is None:
        return df, FEATURES
    transformed = preprocessor.transform(df)
    try:
        feature_names = preprocessor.get_feature_names_out(FEATURES)
    except Exception:
        feature_names = [f"f{i}" for i in range(transformed.shape[1])]
    return transformed, feature_names

def _is_tree_model(estimator):
    name = estimator.__class__.__name__.lower()
    return any(token in name for token in [
        "xgb", "lgbm", "catboost", "randomforest", "gradientboost", "xgboost"
    ])

def _predict_proba_fn(model):
    def _fn(raw_np):
        df = pd.DataFrame(raw_np, columns=FEATURES)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df[FEATURES])
        fallback = _pipeline_fallback_predict(model, df[FEATURES])
        if fallback is not None:
            _, probas = fallback
            if probas is None:
                return np.column_stack([1 - fallback[0], fallback[0]])
            return np.column_stack([1 - probas, probas])
        preds = model.predict(df[FEATURES])
        return np.column_stack([1 - preds, preds])
    return _fn

def _predict_proba_fn_transformed(model):
    estimator = _get_estimator(model)
    def _fn(transformed_np):
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(transformed_np)
        preds = estimator.predict(transformed_np)
        return np.column_stack([1 - preds, preds])
    return _fn

def generate_background_samples(n=200):
    rows = [generate_demo_input() for _ in range(n)]
    return pd.DataFrame(rows)[FEATURES]

# Main app
def main():
    # Header
    st.markdown("""
        <div class="hero">
            <h1>🚖 Taxi Service Acceptance Prediction</h1>
            <p>Predict whether a driver will accept a trip and understand key drivers behind the decision.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This application uses machine learning to predict "
            "driver acceptance for taxi trip requests based on trip attributes."
        )
        
        st.markdown("---")
        st.markdown("### 🤖 Select Model")
        
        # Model selection
        model_choice = st.selectbox(
            "Choose ML Model",
            options=list(MODEL_FILES.keys()),
            help="Select the machine learning model for prediction"
        )
        
        # Load selected model
        model = load_model(model_choice)
        
        st.markdown("### 📊 Model Information")
        st.markdown("""
            <div class="metric-grid">
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card"><h4>Model</h4><span>{model_choice.replace('_', ' ').title()}</span></div>
            <div class="metric-card"><h4>Features</h4><span>{len(FEATURES)}</span></div>
            <div class="metric-card"><h4>Output</h4><span>Accept / Reject</span></div>
            <div class="metric-card"><h4>Status</h4><span>{'Active ✅' if model else 'Demo 🟡'}</span></div>
        """, unsafe_allow_html=True)
        st.markdown("""
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top:0;">🚦 Key Inputs</h3>
                <ul style="margin-bottom:0;">
                    <li>Trip fare</li>
                    <li>Cash or card</li>
                    <li>Batch trip</li>
                    <li>Estimated speed</li>
                    <li>Fare per minute</li>
                    <li>Driver acceptance history</li>
                    <li>Geo encoding</li>
                    <li>Distance bucket</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💬 LLM Recommendations")
        enable_llm = st.checkbox(
            "Enable LLM recommendations",
            value=False,
            help="Uses GROQ_API_KEY to generate suggestions when rejection is predicted."
        )
    
    st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">Trip Inputs</h3>
            <p style="margin-bottom:0; color:#64748b;">Provide trip and driver attributes below, then run the prediction.</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Initialize defaults once
    st.session_state.setdefault('trip_fare_in_lkr', 500.0)
    st.session_state.setdefault('is_cash_trip', 0)
    st.session_state.setdefault('is_batch', 0)
    st.session_state.setdefault('est_speed', 25.0)
    st.session_state.setdefault('fare_per_min_calc', 20.0)
    st.session_state.setdefault('driver_historical_acc_rate', 0.65)
    st.session_state.setdefault('geo_encoded', 10)
    st.session_state.setdefault('dist_ordinal', 2)

    # Demo action (must run before widgets render)
    st.button("🎲 Demo Random Input", use_container_width=True, on_click=apply_demo_values)
    
    with col1:
        st.subheader("🧾 Trip Details")
        
        trip_fare_in_lkr = st.number_input(
            "Trip Fare (LKR)",
            min_value=0.0,
            step=10.0,
            key="trip_fare_in_lkr",
            help="Estimated trip fare in LKR"
        )
        
        is_cash_trip = st.selectbox(
            "Cash Trip",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="is_cash_trip",
            help="Is this a cash payment trip?"
        )
        
        is_batch = st.selectbox(
            "Batch Trip",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="is_batch",
            help="Is this a batch/pooled trip?"
        )
        
        est_speed = st.number_input(
            "Estimated Speed (km/h)",
            min_value=0.0,
            step=0.5,
            key="est_speed",
            help="Estimated trip speed"
        )
    
    with col2:
        st.subheader("📈 Driver & Route Signals")
        
        fare_per_min_calc = st.number_input(
            "Fare per Minute (LKR/min)",
            min_value=0.0,
            step=0.5,
            key="fare_per_min_calc",
            help="Calculated fare per minute"
        )
        
        driver_historical_acc_rate = st.number_input(
            "Driver Historical Acceptance Rate",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="driver_historical_acc_rate",
            help="Driver historical acceptance rate (0-1)"
        )
        
        geo_encoded = st.number_input(
            "Geo Encoded Value",
            min_value=0,
            step=1,
            key="geo_encoded",
            help="Encoded geo zone (integer)"
        )
        
        dist_ordinal = st.number_input(
            "Distance Ordinal",
            min_value=0,
            step=1,
            key="dist_ordinal",
            help="Distance bucket/ordinal (integer)"
        )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Predict Acceptance", use_container_width=True)

    should_predict = predict_button or st.session_state.get("run_prediction", False)
    
    if should_predict:
        st.session_state["run_prediction"] = False
        # Validate inputs
        if trip_fare_in_lkr <= 0:
            st.error("❌ Please enter a valid trip fare.")
            return
        
        if est_speed <= 0:
            st.error("❌ Please enter a valid estimated speed.")
            return
        
        if fare_per_min_calc <= 0:
            st.error("❌ Please enter a valid fare per minute.")
            return
        
        # Prepare input data
        input_data = {
            'trip_fare_in_lkr': trip_fare_in_lkr,
            'is_cash_trip': is_cash_trip,
            'is_batch': is_batch,
            'est_speed': est_speed,
            'fare_per_min_calc': fare_per_min_calc,
            'driver_historical_acc_rate': driver_historical_acc_rate,
            'geo_encoded': geo_encoded,
            'dist_ordinal': dist_ordinal
        }
        
        # Make prediction
        with st.spinner("Analyzing trip request..."):
            prediction, probability = predict_acceptance(input_data, model)
        
        # Display results
        st.markdown("## 📋 Prediction Results")

        # Influence hints
        signals = []
        if driver_historical_acc_rate < 0.4:
            signals.append("Low driver historical acceptance rate")
        if fare_per_min_calc < 10:
            signals.append("Low fare per minute")
        if is_batch == 1:
            signals.append("Batch trip flag")
        if dist_ordinal >= 4:
            signals.append("Long-distance bucket")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box accept">
                    <h2>✅ ACCEPTED</h2>
                    <h3>Acceptance Probability: {probability*100:.1f}%</h3>
                    <p>This trip is likely to be accepted by a driver.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ **Recommendation**: Proceed with dispatching this request.")
            
        else:
            st.markdown(f"""
                <div class="prediction-box reject">
                    <h2>❌ REJECTED</h2>
                    <h3>Acceptance Probability: {probability*100:.1f}%</h3>
                    <p>This trip is less likely to be accepted by a driver.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ **Recommendation**: Consider incentive, reroute, or adjust pricing.")

            if enable_llm:
                with st.spinner("Generating recommendations..."):
                    llm_text = generate_recommendation(input_data, signals)
                st.markdown("### 💡 LLM Recommendations")
                st.info(llm_text)
        
        # Display input summary
        with st.expander("📊 View Input Summary"):
            summary_df = pd.DataFrame([input_data]).T
            summary_df.columns = ['Value']
            st.dataframe(summary_df, use_container_width=True)
        
        # Influence hints
        st.markdown("### 🔍 Notable Signals")
        if signals:
            st.warning("**Potential Impact Signals:**")
            for signal in signals:
                st.markdown(f"- ⚠️ {signal}")
        else:
            st.info("No major negative signals detected for this trip.")

        st.markdown("---")
        st.markdown("## 📊 Model Analytics")
        analytics_tabs = st.tabs(["Local SHAP", "Local LIME", "Global SHAP"])

        with analytics_tabs[0]:
            if model is None:
                st.warning("Model analytics are unavailable in demo mode.")
            elif shap is None or plt is None:
                st.warning("SHAP analytics require shap and matplotlib packages.")
            else:
                try:
                    estimator = _get_estimator(model)
                    if not _is_tree_model(estimator):
                        st.warning("Local SHAP is optimized for tree-based models.")
                    else:
                        sample_df = pd.DataFrame([input_data])[FEATURES]
                        transformed, feature_names = _transform_if_needed(model, sample_df)
                        explainer = shap.TreeExplainer(estimator)
                        shap_values = explainer.shap_values(transformed)
                        base_value = explainer.expected_value
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                            base_value = base_value[1] if isinstance(base_value, list) else base_value
                        exp = shap.Explanation(
                            values=shap_values[0],
                            base_values=base_value,
                            data=transformed[0],
                            feature_names=feature_names,
                        )
                        shap.plots.waterfall(exp, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                except Exception as exc:
                    st.warning(f"Unable to render SHAP explanation. Details: {exc}")

        with analytics_tabs[1]:
            if model is None:
                st.warning("Model analytics are unavailable in demo mode.")
            elif LimeTabularExplainer is None or plt is None:
                st.warning("LIME analytics require lime and matplotlib packages.")
            else:
                try:
                    background = generate_background_samples(200)
                    transformed_bg, feature_names = _transform_if_needed(model, background)
                    explainer = LimeTabularExplainer(
                        training_data=np.asarray(transformed_bg),
                        feature_names=list(feature_names),
                        class_names=["Reject", "Accept"],
                        mode="classification",
                    )
                    sample_df = pd.DataFrame([input_data])[FEATURES]
                    transformed_sample, _ = _transform_if_needed(model, sample_df)
                    sample = np.asarray(transformed_sample)[0]
                    exp = explainer.explain_instance(
                        data_row=sample,
                        predict_fn=_predict_proba_fn_transformed(model) if _get_preprocessor(model) is not None else _predict_proba_fn(model),
                        num_features=8,
                    )
                    st.pyplot(exp.as_pyplot_figure(), clear_figure=True)
                    st.markdown("**Top Factors**")
                    for name, weight in exp.as_list():
                        st.markdown(f"- {name}: {weight:+.3f}")
                except Exception as exc:
                    st.warning("Unable to render LIME explanation.")
                    st.caption(f"Details: {exc}")

        with analytics_tabs[2]:
            if model is None:
                st.warning("Model analytics are unavailable in demo mode.")
            elif shap is None or plt is None:
                st.warning("SHAP analytics require shap and matplotlib packages.")
            else:
                try:
                    estimator = _get_estimator(model)
                    if not _is_tree_model(estimator):
                        st.warning("Global SHAP is optimized for tree-based models.")
                    else:
                        background = generate_background_samples(200)
                        transformed_bg, feature_names = _transform_if_needed(model, background)
                        explainer = shap.TreeExplainer(estimator)
                        shap_values = explainer.shap_values(transformed_bg)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        shap.summary_plot(shap_values, transformed_bg, feature_names=feature_names, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                except Exception as exc:
                    st.warning("Unable to render global SHAP summary.")
                    st.caption(f"Details: {exc}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 1rem;'>
            <p><b>Disclaimer:</b> This tool is for educational and analytical purposes only.</p>
            <p>Predictions are probabilistic and should be used alongside operational judgment.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
