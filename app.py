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

# Custom CSS for a cleaner, more modern look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    .main { 
        background-color: #fafbfc; 
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%);
        color: white; 
        font-size: 16px; 
        font-weight: 600; 
        padding: 14px 24px; 
        border-radius: 12px; 
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4);
    }
    
    /* Example buttons */
    div[data-testid="stHorizontalBlock"] .stButton>button {
        background: #ffffff;
        color: #1e3a5f;
        font-size: 14px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 2px solid #d0d7de;
        font-weight: 500;
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:hover {
        background: #f0f4f8;
        border-color: #2d5a87;
        transform: none;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:active,
    div[data-testid="stHorizontalBlock"] .stButton>button:focus {
        background: #2d5a87 !important;
        color: white !important;
        border-color: #1e3a5f !important;
        box-shadow: 0 0 0 3px rgba(45, 90, 135, 0.3) !important;
    }
    
    /* Result cards */
    .result-card { 
        padding: 28px; 
        border-radius: 16px; 
        text-align: center; 
        margin-top: 24px; 
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    .fraud { 
        background: linear-gradient(135deg, #fff5f5 0%, #ffe3e3 100%);
        border-left: 6px solid #e53e3e;
    }
    .fraud h2 { color: #c53030; }
    .fraud p { color: #742a2a; }
    
    .safe { 
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 6px solid #38a169;
    }
    .safe h2 { color: #276749; }
    .safe p { color: #22543d; }
    
    /* Section headers */
    h4 {
        color: #1e3a5f !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Captions styling */
    .stCaption {
        color: #6b7c93 !important;
        margin-top: -0.5rem !important;
    }
    
    /* Divider styling */
    hr {
        margin: 1.5rem 0 !important;
        border-color: #e2e8f0 !important;
    }
    
    /* Explanation box */
    .explanation-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        margin-top: 16px;
        border: 1px solid #e2e8f0;
    }
    .explanation-box h4 {
        margin: 0 0 8px 0;
        color: #475569;
        font-size: 14px;
    }
    .explanation-box p {
        margin: 0;
        color: #64748b;
        font-size: 13px;
        line-height: 1.5;
    }
    
    /* Info expander */
    .streamlit-expanderHeader {
        background: #f0f4f8;
        border-radius: 10px;
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve number input styling */
    .stNumberInput input {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with Hugging Face Hub support
@st.cache_resource
def load_model():
    """Load model from Hugging Face Hub"""
    try:
        from huggingface_hub import hf_hub_download
        
        # Get token from secrets or env
        hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        
        if not hf_token:
            st.warning("⚠️ HF_TOKEN not found in secrets. Trying to load without authentication...")
        
        HF_REPO = os.environ.get("HF_MODEL_REPO", "rahulmauryaa/Fraud-detection-model")
        
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="best_model_Recall.pkl",
            repo_type="model",
            token=hf_token
        )
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded from Hugging Face Hub")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model from Hugging Face: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

model = load_model()

# ===== HEADER =====
st.markdown("""
    <div class="app-header">
        <h1>🛡️ FraudGuard AI</h1>
        <p>Instant fraud detection for financial transactions</p>
    </div>
""", unsafe_allow_html=True)

# ===== HOW TO USE EXPANDER =====
with st.expander("ℹ️ How to use this tool", expanded=False):
    st.markdown("""
    **Quick Start:**
    1. Select the **transaction type** (e.g., CASH_OUT, TRANSFER)
    2. Enter the **amount** and **account balances** before & after the transaction
    3. Click **"Check if Fraudulent"** to get instant results
    
    **What does the model check?**  
    This AI analyzes patterns like unusual amounts, balance mismatches, and risky transaction types 
    to detect potential fraud. It was trained on millions of real transactions.
    
    **Not sure what to enter?** Try the example buttons below to see how it works!
    """)

# ===== EXAMPLE TRANSACTIONS =====
st.markdown("##### Try an Example")
col_ex1, col_ex2 = st.columns(2)

# Initialize session state for form values
if 'example_loaded' not in st.session_state:
    st.session_state.example_loaded = None
    st.session_state.type_val = 'CASH_OUT'
    st.session_state.amount = 1000.0
    st.session_state.step = 1
    st.session_state.oldbalanceOrg = 5000.0
    st.session_state.newbalanceOrig = 4000.0
    st.session_state.oldbalanceDest = 1000.0
    st.session_state.newbalanceDest = 2000.0

with col_ex1:
    if st.button("✅ Normal Transaction", use_container_width=True, key="btn_normal"):
        st.session_state.type_val = 'PAYMENT'
        st.session_state.amount = 500.0
        st.session_state.step = 1
        st.session_state.oldbalanceOrg = 10000.0
        st.session_state.newbalanceOrig = 9500.0
        st.session_state.oldbalanceDest = 2000.0
        st.session_state.newbalanceDest = 2500.0
        st.session_state.example_loaded = 'normal'
        st.rerun()

with col_ex2:
    if st.button("🚨 Suspicious/Fraud", use_container_width=True, key="btn_fraud"):
        st.session_state.type_val = 'TRANSFER'
        st.session_state.amount = 1810000.0
        st.session_state.step = 1
        st.session_state.oldbalanceOrg = 5000000.0
        st.session_state.newbalanceOrig = 0.0
        st.session_state.oldbalanceDest = 1000.0
        st.session_state.newbalanceDest = 2000.0
        st.session_state.example_loaded = 'suspicious'
        st.rerun()

st.write("")  # Spacing

# ===== INPUT FORM =====
with st.form("prediction_form"):
    
    # ----- Transaction Details Section -----
    st.markdown("#### 💳 Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        type_options = ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT']
        type_val = st.selectbox(
            "Transaction Type",
            type_options,
            index=type_options.index(st.session_state.type_val),
            help="type of transaction"
        )
        st.caption("How money is being moved ")
    
    with col2:
        amount = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.0, 
            max_value=10000000.0,
            value=st.session_state.amount, 
            step=100.0, 
            format="%.2f",
            help="The amount being transferred in this transaction"
        )
        st.caption("Typical range: 100 - 50,000")
    
    st.divider()
    
    # ----- Sender & Receiver Section -----
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("##### Sender's Account")
        
        oldbalanceOrg = st.number_input(
            "Sender's Balance Before Transaction ($)", 
            min_value=0.0, 
            max_value=50000000.0,
            value=st.session_state.oldbalanceOrg, 
            step=100.0,
            format="%.2f",
            key="org_old"
        )
        
        newbalanceOrig = st.number_input(
            "Balance After Transaction ($)", 
            min_value=0.0, 
            max_value=50000000.0,
            value=st.session_state.newbalanceOrig, 
            step=100.0,
            format="%.2f",
            key="org_new"
        )

    with col4:
        st.markdown("##### Receiver Account")
        
        oldbalanceDest = st.number_input(
            "Receiver's Balance Before Transaction ($)", 
            min_value=0.0, 
            max_value=50000000.0,
            value=st.session_state.oldbalanceDest, 
            step=100.0,
            format="%.2f",
            key="dest_old"
        )        
        newbalanceDest = st.number_input(
            "Balance After Transaction ($)", 
            min_value=0.0, 
            max_value=50000000.0,
            value=st.session_state.newbalanceDest, 
            step=100.0,
            format="%.2f",
            key="dest_new"
        )
    
    
    step = st.session_state.step
    isFlaggedFraud = 0

    # Submit button
    st.write("")  # Spacing
    submit_btn = st.form_submit_button("🔍 Check if Fraudulent", use_container_width=True)

# ===== PREDICTION LOGIC =====
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
            # Get prediction probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]  # Probability of fraud
            else:
                raw_pred = model.predict(input_df)[0]
                proba = 1.0 if raw_pred == 1 else 0.0
            
            # Use lower threshold for fraud detection (more sensitive)
            FRAUD_THRESHOLD = 0.3
            prediction = 1 if proba >= FRAUD_THRESHOLD else 0

            # Generate explanation based on transaction features
            explanations = []
            
            if type_val in ['TRANSFER', 'CASH_OUT'] and amount > 100000:
                explanations.append(f"Large {type_val} of ${amount:,.0f}")
            
            if oldbalanceOrg > 0 and newbalanceOrig == 0:
                explanations.append("Account fully emptied")
            
            if type_val in ['TRANSFER', 'CASH_OUT'] and oldbalanceDest == 0:
                explanations.append("Receiver had zero balance")
            
            if amount > oldbalanceOrg:
                explanations.append("Amount exceeds available balance")
            
            if type_val == 'TRANSFER' and newbalanceDest == 0 and amount > 0:
                explanations.append("Transfer didn't increase receiver balance")
            
            if not explanations:
                if prediction == 1:
                    explanations.append("Pattern matches known fraud indicators")
                else:
                    explanations.append("Transaction follows normal patterns")

            # Display results
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-card fraud">
                        <h2>🚨 High Risk — Likely Fraudulent</h2>
                        <p style="font-size: 22px; margin: 16px 0;">
                            Fraud Probability: <strong>{proba:.1%}</strong>
                        </p>
                        <p style="font-size: 15px; opacity: 0.9;">
                            Recommended Action: <strong>Block & Review Manually</strong>
                        </p>
                    </div>
                    <div class="explanation-box">
                        <h4>💡 Why this might be fraud:</h4>
                        <p>{' • '.join(explanations)}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card safe">
                        <h2>✅ Low Risk — Appears Legitimate</h2>
                        <p style="font-size: 22px; margin: 16px 0;">
                            Fraud Probability: <strong>{proba:.1%}</strong>
                        </p>
                        <p style="font-size: 15px; opacity: 0.9;">
                            Recommended Action: <strong>Safe to Process</strong>
                        </p>
                    </div>
                    <div class="explanation-box">
                        <h4>💡 Analysis summary:</h4>
                        <p>{' • '.join(explanations)}</p>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.error("Model is not loaded. Please restart the application.")

# ===== FOOTER =====
st.write("")
st.write("")
st.caption("Built with ML • Trained on 6M+ transactions • Balanced Random Forest Model")
