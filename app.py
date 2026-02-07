import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('Models/lr_fraud_detector.pkl')
        rf_model = joblib.load('Models/rf_fraud_detector.pkl')
        scaler = joblib.load('Models/scaler.pkl')
        return lr_model, rf_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

lr_model, rf_model, scaler = load_models()

# Title
st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Performance"])

# Home Page
if page == "🏠 Home":
    st.markdown("## Welcome to the Fraud Detection System!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Size", "284,807 transactions")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Rate", "0.173%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Model Recall", "87.84%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Project Overview")
    st.write("""
    This application uses machine learning to detect fraudulent credit card transactions.
    The models were trained on a highly imbalanced dataset with only 0.173% fraudulent transactions.
    
    **Key Features:**
    - ✅ Multiple ML models (Logistic Regression, Random Forest)
    - ✅ SMOTE technique to handle class imbalance
    - ✅ High recall rate (87.84%) to minimize missed frauds
    - ✅ Real-time predictions on new transactions
    """)
    
    st.markdown("### 📊 Model Performance Summary")
    
    performance_data = {
        'Model': ['LR + SMOTE', 'Random Forest', 'XGBoost'],
        'Accuracy': ['97.73%', '99.81%', '99.72%'],
        'Recall': ['87.84%', '81.08%', '83.78%'],
        'Precision': ['6%', '47%', '36%']
    }
    
    df_performance = pd.DataFrame(performance_data)
    st.table(df_performance)
    
    st.info("💡 **Note:** Higher recall means we catch more frauds, but may have more false alarms.")

# Single Prediction Page
elif page == "🔍 Single Prediction":
    st.markdown("## 🔍 Single Transaction Prediction")
    st.write("Enter transaction details to check if it's fraudulent:")
    
    # Model selection
    model_choice = st.selectbox("Select Model", ["Logistic Regression + SMOTE (Best Recall)", "Random Forest (Best Balance)"])
    
    st.markdown("---")
    
    # Input method
    input_method = st.radio("Input Method", ["Manual Entry", "Random Sample"])
    
    if input_method == "Manual Entry":
        st.markdown("### Enter Transaction Features:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)
            v10 = st.number_input("V10", value=0.0)
            v11 = st.number_input("V11", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v13 = st.number_input("V13", value=0.0)
        
        with col2:
            v14 = st.number_input("V14", value=0.0)
            v15 = st.number_input("V15", value=0.0)
            v16 = st.number_input("V16", value=0.0)
            v17 = st.number_input("V17", value=0.0)
            v18 = st.number_input("V18", value=0.0)
            v19 = st.number_input("V19", value=0.0)
            v20 = st.number_input("V20", value=0.0)
            v21 = st.number_input("V21", value=0.0)
            v22 = st.number_input("V22", value=0.0)
            v23 = st.number_input("V23", value=0.0)
            v24 = st.number_input("V24", value=0.0)
            v25 = st.number_input("V25", value=0.0)
            v26 = st.number_input("V26", value=0.0)
            v27 = st.number_input("V27", value=0.0)
            v28 = st.number_input("V28", value=0.0)
        
        # Create feature array
        features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                             v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]])
        
        # Scale time and amount
        scaled_time = scaler.transform([[time]])[0][0] if scaler else time
        scaled_amount = scaler.transform([[amount]])[0][0] if scaler else amount
        
        # Combine all features
        final_features = np.concatenate([features, [[scaled_amount, scaled_time]]], axis=1)
    
    else:  # Random Sample
        st.info("📌 Generating a random sample transaction...")
        # Generate random features
        features = np.random.randn(1, 28)
        time = np.random.uniform(0, 172792)
        amount = np.random.uniform(0, 500)
        
        scaled_time = scaler.transform([[time]])[0][0] if scaler else time
        scaled_amount = scaler.transform([[amount]])[0][0] if scaler else amount
        
        final_features = np.concatenate([features, [[scaled_amount, scaled_time]]], axis=1)
        
        st.write(f"**Amount:** ${amount:.2f}")
        st.write(f"**Time:** {time:.2f} seconds")
    
    # Predict button
    if st.button("🔮 Predict", type="primary"):
        if model_choice == "Logistic Regression + SMOTE (Best Recall)":
            model = lr_model
        else:
            model = rf_model
        
        if model:
            # Make prediction
            prediction = model.predict(final_features)[0]
            probability = model.predict_proba(final_features)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **FRAUDULENT TRANSACTION DETECTED!**")
                    st.markdown(f"**Fraud Probability:** {probability[1]*100:.2f}%")
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION**")
                    st.markdown(f"**Legitimate Probability:** {probability[0]*100:.2f}%")
            
            with col2:
                # Probability visualization
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Legitimate', 'Fraud']
                colors = ['#66b3ff', '#ff6b6b']
                ax.bar(labels, probability, color=colors)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Confidence')
                ax.set_ylim([0, 1])
                for i, v in enumerate(probability):
                    ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
                st.pyplot(fig)

# Batch Prediction Page
elif page == "📊 Batch Prediction":
    st.markdown("## 📊 Batch Prediction")
    st.write("Upload a CSV file with multiple transactions to get predictions:")
    
    # Model selection
    model_choice = st.selectbox("Select Model", ["Logistic Regression + SMOTE", "Random Forest"])
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} transactions found.")
            
            st.markdown("### Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🔮 Predict All", type="primary"):
                model = lr_model if model_choice == "Logistic Regression + SMOTE" else rf_model
                
                if model:
                    # Make predictions
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)
                    
                    # Add results to dataframe
                    df['Prediction'] = predictions
                    df['Fraud_Probability'] = probabilities[:, 1]
                    df['Result'] = df['Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Legitimate')
                    
                    st.markdown("---")
                    st.markdown("### 📋 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    fraud_count = (predictions == 1).sum()
                    legitimate_count = (predictions == 0).sum()
                    
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        st.metric("Fraudulent", fraud_count, delta=f"{fraud_count/len(df)*100:.2f}%")
                    with col3:
                        st.metric("Legitimate", legitimate_count, delta=f"{legitimate_count/len(df)*100:.2f}%")
                    
                    st.markdown("### Results Table:")
                    st.dataframe(df[['Prediction', 'Fraud_Probability', 'Result']].head(20))
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Model Performance Page
else:
    st.markdown("## 📈 Model Performance Analysis")
    
    st.markdown("### 🎯 Key Metrics Comparison")
    
    # Performance table
    performance_data = {
        'Model': ['Logistic Regression (Baseline)', 'LR + SMOTE', 'Random Forest', 'XGBoost'],
        'Accuracy (%)': [99.92, 97.73, 99.81, 99.72],
        'Recall (%)': [61.49, 87.84, 81.08, 83.78],
        'Precision (%)': [86, 6, 47, 36],
        'Frauds Caught': [91, 130, 120, 124],
        'Frauds Missed': [57, 18, 28, 24]
    }
    
    df_perf = pd.DataFrame(performance_data)
    st.dataframe(df_perf, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Recall Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax.barh(df_perf['Model'], df_perf['Recall (%)'], color=colors)
        ax.set_xlabel('Recall (%)')
        ax.set_title('Model Recall (Fraud Detection Rate)')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📊 Frauds Caught vs Missed")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df_perf))
        width = 0.35
        ax.bar(x - width/2, df_perf['Frauds Caught'], width, label='Caught', color='#66b3ff')
        ax.bar(x + width/2, df_perf['Frauds Missed'], width, label='Missed', color='#ff9999')
        ax.set_ylabel('Number of Frauds')
        ax.set_title('Frauds Caught vs Missed (out of 148)')
        ax.set_xticks(x)
        ax.set_xticklabels(df_perf['Model'], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.markdown("### 💡 Key Insights")
    st.info("""
    - **LR + SMOTE** achieves the highest recall (87.84%), catching 130 out of 148 frauds
    - **Random Forest** offers the best balance with 99.81% accuracy and 81% recall
    - **SMOTE technique** significantly improved fraud detection from 61% to 88% recall
    - Trade-off: Higher recall comes with lower precision (more false alarms)
    - In fraud detection, catching frauds is more critical than false positives
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Credit Card Fraud Detection ML Project</p>
    </div>
""", unsafe_allow_html=True)