import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stButton>button { width: 100%; background-color: #4CAF50; color: white; font-size: 18px; font-weight: bold; padding: 10px; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #45a049; }
    .result-card { padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); }
    .fraud { background-color: #ffebee; color: #d32f2f; border-left: 10px solid #d32f2f; }
    .safe { background-color: #e8f5e9; color: #2e7d32; border-left: 10px solid #2e7d32; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #333; }
    .input-section { background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# Load model with debug
@st.cache_resource
def load_model():
    model_path = 'model/best_model_Recall.pkl'
    
    # Debug info
    st.sidebar.write(f"CWD: {os.getcwd()}")
    st.sidebar.write(f"Model Path: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file missing at {os.path.abspath(model_path)}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        # Print detailed error to console
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Header
st.title("🛡️ FraudGuard AI")
st.markdown("### Real-time Transaction Analysis")
st.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.subheader("📝 Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### General Info")
        type_options = ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT']
        type_val = st.selectbox("Transaction Type", type_options)
        amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=10.0, format="%.2f")
    
    with col2:
        st.markdown("## Time Step")
        step = st.number_input("Seconds since start", min_value=1, value=1, help="Time step of the transaction")

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 📤 Sender (Origin)")
        oldbalanceOrg = st.number_input("Initial Balance ($)", min_value=0.0, value=1000.0, key="org_old")
        newbalanceOrig = st.number_input("New Balance ($)", min_value=0.0, value=0.0, key="org_new")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 📥 Receiver (Dest)")
        oldbalanceDest = st.number_input("Initial Balance ($)", min_value=0.0, value=0.0, key="dest_old")
        newbalanceDest = st.number_input("New Balance ($)", min_value=0.0, value=0.0, key="dest_new")
        st.markdown('</div>', unsafe_allow_html=True)

    # Hidden/Default Inputs
    isFlaggedFraud = 0 

    submit_btn = st.form_submit_button("🔍 Analyze Transaction")

# Prediction Logic
if submit_btn:
    if model:
        # Preprocessing
        type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        type_encoded = type_mapping.get(type_val, 0)

        input_df = pd.DataFrame({
            'step': [step],
            'type': [type_encoded],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
            'isFlaggedFraud': [isFlaggedFraud]
        })

        try:
            # Get prediction
            prediction = model.predict(input_df)[0]
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]
            else:
                proba = 1.0 if prediction == 1 else 0.0

            if prediction == 1:
                st.markdown(f"""
                    <div class="result-card fraud">
                        <h2>🚨 High Risk Transaction Detected</h2>
                        <p style="font-size: 20px;">Fraud Probability: <strong>{proba:.1%}</strong></p>
                        <p>Action Recommended: <strong>Block & Review</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card safe">
                        <h2>✅ Transaction Appears Safe</h2>
                        <p style="font-size: 20px;">Fraud Probability: <strong>{proba:.1%}</strong></p>
                        <p>Action Recommended: <strong>Process</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.error("Model is not loaded. Please restart the application.")
