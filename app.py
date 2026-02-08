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
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .action-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #000000;
    }
    .action-card h4 {
        color: #000000;
        margin-top: 0;
    }
    .action-card li, .action-card p, .action-card strong {
        color: #000000;
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
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None

lr_model, rf_model, scaler = load_models()

# Title
st.markdown('<h1 class="main-header">🔒 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("### 📊 Navigation")
page = st.sidebar.radio("Select Page", 
                        ["🏠 Home", "🔍 Single Prediction", "📁 Batch Prediction", "📈 Model Performance"],
                        label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**Fraud Detection System**

Built with Machine Learning to identify fraudulent credit card transactions in real-time.

**Key Features:**
- Multiple ML models
- 87.84% fraud detection rate
- Real-time predictions
- Feature importance analysis
""")

# Home Page
if page == "🏠 Home":
    st.markdown("## 📋 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Dataset Size", "284,807", help="Total transactions analyzed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚠️ Fraud Rate", "0.173%", help="Percentage of fraudulent transactions")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("✅ Best Recall", "87.84%", delta="↑ 26.35%", help="Improvement over baseline")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Accuracy", "99.81%", help="Random Forest model accuracy")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.write("""
        This application uses advanced machine learning algorithms to detect fraudulent credit card transactions 
        in real-time. The system was trained on a highly imbalanced dataset with only 0.173% fraudulent transactions.
        
        **🔑 Key Capabilities:**
        - Real-time fraud detection with 87.84% recall rate
        - Multiple ML models (Logistic Regression, Random Forest, XGBoost)
        - SMOTE technique to handle severe class imbalance
        - Feature importance analysis - see which signals contribute to predictions
        - Batch processing for multiple transactions
        - Professional risk assessment and recommendations
        """)
    
    with col2:
        st.markdown("### 📊 Model Performance")
        performance_data = {
            'Model': ['LR + SMOTE', 'Random Forest', 'XGBoost'],
            'Recall': [87.84, 81.08, 83.78],
            'Precision': [6, 47, 36],
            'Accuracy': [97.73, 99.81, 99.72]
        }
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 1️⃣ Data Input")
        st.write("Transaction details are collected and preprocessed")
    
    with col2:
        st.markdown("#### 2️⃣ Analysis")
        st.write("Multiple ML models analyze the transaction")
    
    with col3:
        st.markdown("#### 3️⃣ Prediction")
        st.write("System provides fraud probability and risk level")
    
    with col4:
        st.markdown("#### 4️⃣ Action")
        st.write("Recommended actions based on risk assessment")

# Single Prediction Page
elif page == "🔍 Single Prediction":
    st.markdown("## 🔍 Transaction Analysis & Prediction")
    
    input_method = st.radio("📥 Input Method", ["🎲 Random Sample", "✍️ Manual Entry"], horizontal=True)
    
    st.markdown("---")
    
    if input_method == "✍️ Manual Entry":
        st.markdown("### 📝 Enter Transaction Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("⏱️ Time (seconds)", min_value=0.0, value=0.0)
            amount = st.number_input("💵 Amount ($)", min_value=0.0, value=100.0)
            st.markdown("**🔢 PCA Features (V1-V14)**")
            v1 = st.number_input("V1", value=0.0, format="%.6f")
            v2 = st.number_input("V2", value=0.0, format="%.6f")
            v3 = st.number_input("V3", value=0.0, format="%.6f")
            v4 = st.number_input("V4", value=0.0, format="%.6f")
            v5 = st.number_input("V5", value=0.0, format="%.6f")
            v6 = st.number_input("V6", value=0.0, format="%.6f")
            v7 = st.number_input("V7", value=0.0, format="%.6f")
            v8 = st.number_input("V8", value=0.0, format="%.6f")
            v9 = st.number_input("V9", value=0.0, format="%.6f")
            v10 = st.number_input("V10", value=0.0, format="%.6f")
            v11 = st.number_input("V11", value=0.0, format="%.6f")
            v12 = st.number_input("V12", value=0.0, format="%.6f")
            v13 = st.number_input("V13", value=0.0, format="%.6f")
            v14 = st.number_input("V14", value=0.0, format="%.6f")
        
        with col2:
            st.markdown("**🔢 PCA Features (V15-V28)**")
            v15 = st.number_input("V15", value=0.0, format="%.6f")
            v16 = st.number_input("V16", value=0.0, format="%.6f")
            v17 = st.number_input("V17", value=0.0, format="%.6f")
            v18 = st.number_input("V18", value=0.0, format="%.6f")
            v19 = st.number_input("V19", value=0.0, format="%.6f")
            v20 = st.number_input("V20", value=0.0, format="%.6f")
            v21 = st.number_input("V21", value=0.0, format="%.6f")
            v22 = st.number_input("V22", value=0.0, format="%.6f")
            v23 = st.number_input("V23", value=0.0, format="%.6f")
            v24 = st.number_input("V24", value=0.0, format="%.6f")
            v25 = st.number_input("V25", value=0.0, format="%.6f")
            v26 = st.number_input("V26", value=0.0, format="%.6f")
            v27 = st.number_input("V27", value=0.0, format="%.6f")
            v28 = st.number_input("V28", value=0.0, format="%.6f")
        
        features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                             v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]])
        
        scaled_time = scaler.transform([[time]])[0][0] if scaler else time
        scaled_amount = scaler.transform([[amount]])[0][0] if scaler else amount
        
        final_features = np.concatenate([features, [[scaled_amount, scaled_time]]], axis=1)
    
    else:
        st.info("🎲 Generating random transaction sample...")
        features = np.random.randn(1, 28)
        time = np.random.uniform(0, 172792)
        amount = np.random.uniform(10, 500)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💵 Transaction Amount", f"${amount:.2f}")
        with col2:
            st.metric("⏱️ Time", f"{time:.0f} seconds")
        
        scaled_time = scaler.transform([[time]])[0][0] if scaler else time
        scaled_amount = scaler.transform([[amount]])[0][0] if scaler else amount
        
        final_features = np.concatenate([features, [[scaled_amount, scaled_time]]], axis=1)
        
        v_features = features[0]
        v14, v17, v12, v10, v4 = v_features[13], v_features[16], v_features[11], v_features[9], v_features[3]
    
    st.markdown("---")
    
    if st.button("🔮 Analyze Transaction", type="primary", use_container_width=True):
        if lr_model and rf_model:
            pred_lr = lr_model.predict(final_features)[0]
            prob_lr = lr_model.predict_proba(final_features)[0]
            
            pred_rf = rf_model.predict(final_features)[0]
            prob_rf = rf_model.predict_proba(final_features)[0]
            
            avg_fraud_prob = (prob_lr[1] + prob_rf[1]) / 2
            
            if avg_fraud_prob >= 0.7:
                risk_level = "HIGH"
                risk_class = "risk-high"
                risk_icon = "🔴"
            elif avg_fraud_prob >= 0.3:
                risk_level = "MEDIUM"
                risk_class = "risk-medium"
                risk_icon = "🟡"
            else:
                risk_level = "LOW"
                risk_class = "risk-low"
                risk_icon = "🟢"
            
            st.markdown("---")
            st.markdown(f'<div class="{risk_class}">{risk_icon} RISK LEVEL: {risk_level}</div>', unsafe_allow_html=True)
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📊 Prediction Results")
                
                # Create side-by-side comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Logistic Regression
                colors_prob = ['#10b981', '#ef4444']
                probabilities_lr = [prob_lr[0]*100, prob_lr[1]*100]
                labels = ['Legitimate', 'Fraud']
                
                bars1 = ax1.barh(labels, probabilities_lr, color=colors_prob)
                ax1.set_xlabel('Probability (%)', fontsize=11)
                ax1.set_title(f'Logistic Regression\nFraud: {prob_lr[1]*100:.1f}%', fontsize=12, fontweight='bold')
                ax1.set_xlim([0, 100])
                
                for i, (bar, val) in enumerate(zip(bars1, probabilities_lr)):
                    ax1.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=10)
                
                # Random Forest
                probabilities_rf = [prob_rf[0]*100, prob_rf[1]*100]
                
                bars2 = ax2.barh(labels, probabilities_rf, color=colors_prob)
                ax2.set_xlabel('Probability (%)', fontsize=11)
                ax2.set_title(f'Random Forest\nFraud: {prob_rf[1]*100:.1f}%', fontsize=12, fontweight='bold')
                ax2.set_xlim([0, 100])
                
                for i, (bar, val) in enumerate(zip(bars2, probabilities_rf)):
                    ax2.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show consensus metric
                diff = abs(prob_lr[1] - prob_rf[1]) * 100
                avg_fraud_prob = (prob_lr[1] + prob_rf[1]) / 2
                
                col_consensus1, col_consensus2 = st.columns(2)
                with col_consensus1:
                    st.metric("📊 Average Fraud Probability", f"{avg_fraud_prob*100:.1f}%")
                with col_consensus2:
                    if diff < 10:
                        st.metric("🎯 Model Agreement", "Strong", delta="High Confidence", delta_color="normal")
                    elif diff < 30:
                        st.metric("🎯 Model Agreement", "Moderate", delta="Review Recommended", delta_color="off")
                    else:
                        st.metric("🎯 Model Agreement", "Weak", delta="Manual Review Required", delta_color="inverse")
                
                st.markdown("### 🤖 Model Consensus")
                
                consensus = sum([pred_lr, pred_rf])
                st.markdown(f"**Models Predicting Fraud:** {consensus}/2")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("🔹 Logistic Regression", 
                             "FRAUD" if pred_lr == 1 else "LEGITIMATE",
                             f"{prob_lr[1]*100:.1f}%")
                with col_b:
                    st.metric("🔹 Random Forest", 
                             "FRAUD" if pred_rf == 1 else "LEGITIMATE",
                             f"{prob_rf[1]*100:.1f}%")
                
                if consensus == 2:
                    st.success("✅ Strong consensus: Both models agree")
                elif consensus == 1:
                    st.warning("⚠️ Moderate confidence: Models disagree")
                else:
                    st.info("✅ Both models predict legitimate")
            
            with col2:
                st.markdown("### 🔍 Feature Analysis")
                
                feature_names = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
                feature_values = final_features[0]
                
                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': feature_values,
                        'Importance': importances
                    })
                    
                    top_features = feature_importance_df.nlargest(5, 'Importance')
                    
                    st.markdown("**Top 5 Contributing Features:**")
                    
                    for idx, row in top_features.iterrows():
                        feature = row['Feature']
                        value = row['Value']
                        importance = row['Importance']
                        
                        if feature in ['V14', 'V17', 'V12'] and value < -1:
                            indicator = "⚠️ High Risk"
                            bg_color = "#fee2e2"
                            text_color = "#000000"
                            border_color = "#ef4444"
                        elif abs(value) > 2:
                            indicator = "⚠️ Unusual"
                            bg_color = "#fef3c7"
                            text_color = "#000000"
                            border_color = "#f59e0b"
                        else:
                            indicator = "✅ Normal"
                            bg_color = "#d1fae5"
                            text_color = "#000000"
                            border_color = "#10b981"
                        
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; color: {text_color}; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <strong style="font-size: 1.1em; color: {text_color};">{feature}</strong>: <span style="color: {text_color}; font-weight: 700;">{value:.4f}</span><br>
                            <span style="color: {text_color}; font-size: 0.95em; font-weight: 500;">Importance: {importance:.4f} | {indicator}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("**📋 Transaction Details:**")
                st.write(f"**Amount:** ${amount:.2f}")
                st.write(f"**Time:** {time:.0f} seconds")
                
                time_hours = (time % 86400) / 3600
                if 23 <= time_hours or time_hours <= 5:
                    st.warning("⚠️ **Late night transaction** (High risk period)")
                else:
                    st.info("✅ Normal transaction time")
                
                # Add clarification about risk signals
                st.markdown("---")
                st.info("ℹ️ **Note:** Individual risk indicators (time, amount) are contextual signals. Final risk assessment is based on ML model probability combining all 30 features.")
            
            st.markdown("---")
            st.markdown("### 📝 Recommended Actions")
            
            if risk_level == "HIGH":
                st.markdown("""
                <div class="action-card" style="color: #000000;">
                    <h4 style="color: #000000;">🔴 High Risk - Immediate Action Required</h4>
                    <ol style="color: #000000;">
                        <li style="color: #000000;"><strong style="color: #000000;">🚫 Block Transaction</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">📞 Contact Customer</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">🔒 Freeze Card</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">📧 Send Alert</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">📊 Manual Review</strong></li>
                    </ol>
                    <p style="color: #000000;"><strong style="color: #000000;">💰 Estimated Loss if Fraud:</strong> ${:.2f}</p>
                </div>
                """.format(amount), unsafe_allow_html=True)
            elif risk_level == "MEDIUM":
                st.markdown("""
                <div class="action-card" style="color: #000000;">
                    <h4 style="color: #000000;">🟡 Medium Risk - Verification Recommended</h4>
                    <ol style="color: #000000;">
                        <li style="color: #000000;"><strong style="color: #000000;">✉️ Send Verification</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">⏸️ Delay Processing</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">👁️ Monitor Activity</strong></li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="action-card" style="background: #d1fae5; border-left-color: #10b981; color: #000000;">
                    <h4 style="color: #000000;">🟢 Low Risk - Safe to Process</h4>
                    <ol style="color: #000000;">
                        <li style="color: #000000;"><strong style="color: #000000;">✅ Approve Transaction</strong></li>
                        <li style="color: #000000;"><strong style="color: #000000;">📊 Standard Monitoring</strong></li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

# Batch Prediction Page
elif page == "📁 Batch Prediction":
    st.markdown("## 📁 Batch Transaction Analysis")
    
    model_choice = st.selectbox("🤖 Select Model", 
                                ["Logistic Regression + SMOTE", "Random Forest"])
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! Found **{len(df):,}** transactions")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🔮 Analyze All", type="primary", use_container_width=True):
                model = lr_model if "Logistic" in model_choice else rf_model
                
                if model:
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)
                    
                    df['Prediction'] = predictions
                    df['Fraud_Probability'] = probabilities[:, 1] * 100
                    df['Risk_Level'] = df['Fraud_Probability'].apply(
                        lambda x: 'HIGH' if x >= 70 else ('MEDIUM' if x >= 30 else 'LOW')
                    )
                    df['Result'] = df['Prediction'].apply(lambda x: 'FRAUD' if x == 1 else 'LEGITIMATE')
                    
                    st.success("✅ Analysis complete!")
                    
                    fraud_count = (predictions == 1).sum()
                    legitimate_count = (predictions == 0).sum()
                    high_risk_count = (df['Risk_Level'] == 'HIGH').sum()
                    medium_risk_count = (df['Risk_Level'] == 'MEDIUM').sum()
                    low_risk_count = (df['Risk_Level'] == 'LOW').sum()
                    
                    st.markdown("### 📊 Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📋 Total Transactions", len(df))
                    with col2:
                        st.metric("🔴 Fraudulent", fraud_count, 
                                 delta=f"{fraud_count/len(df)*100:.2f}%")
                    with col3:
                        st.metric("🟢 Legitimate", legitimate_count,
                                 delta=f"{legitimate_count/len(df)*100:.2f}%")
                    with col4:
                        avg_fraud_prob = df['Fraud_Probability'].mean()
                        st.metric("📊 Avg Fraud Prob", f"{avg_fraud_prob:.2f}%")
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.markdown("#### 🎯 Prediction Distribution")
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        
                        # Pie chart for predictions
                        sizes = [legitimate_count, fraud_count]
                        labels = ['Legitimate', 'Fraudulent']
                        colors = ['#10b981', '#ef4444']
                        explode = (0, 0.1)
                        
                        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
                        ax1.set_title(f'Transaction Classification\n({len(df)} total)', 
                                     fontsize=13, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig1)
                    
                    with col_viz2:
                        st.markdown("#### ⚡ Risk Level Distribution")
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        
                        # Bar chart for risk levels
                        risk_data = [high_risk_count, medium_risk_count, low_risk_count]
                        risk_labels = ['HIGH', 'MEDIUM', 'LOW']
                        risk_colors = ['#ef4444', '#f59e0b', '#10b981']
                        
                        bars = ax2.bar(risk_labels, risk_data, color=risk_colors, alpha=0.8)
                        ax2.set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
                        ax2.set_title('Risk Level Breakdown', fontsize=13, fontweight='bold')
                        ax2.grid(alpha=0.3, axis='y')
                        
                        # Add value labels on bars
                        for bar, val in zip(bars, risk_data):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(val)}\n({val/len(df)*100:.1f}%)',
                                    ha='center', va='bottom', fontweight='bold', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                    
                    # Fraud probability distribution
                    st.markdown("---")
                    st.markdown("#### 📈 Fraud Probability Distribution")
                    
                    fig3, ax3 = plt.subplots(figsize=(14, 5))
                    
                    # Histogram of fraud probabilities
                    ax3.hist(df['Fraud_Probability'], bins=50, color='#3b82f6', 
                            alpha=0.7, edgecolor='black')
                    ax3.axvline(x=30, color='#f59e0b', linestyle='--', linewidth=2, 
                               label='Medium Risk Threshold (30%)')
                    ax3.axvline(x=70, color='#ef4444', linestyle='--', linewidth=2, 
                               label='High Risk Threshold (70%)')
                    ax3.set_xlabel('Fraud Probability (%)', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
                    ax3.set_title('Distribution of Fraud Probabilities', fontsize=13, fontweight='bold')
                    ax3.legend(fontsize=10)
                    ax3.grid(alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Results")
                    
                    st.dataframe(df[['Result', 'Risk_Level', 'Fraud_Probability']].head(20))
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fraud_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Model Performance Page
else:
    st.markdown("## 📈 Model Performance Analysis")
    
    performance_data = {
        'Model': ['LR (Baseline)', 'LR + SMOTE', 'Random Forest', 'XGBoost'],
        'Accuracy (%)': [99.92, 97.73, 99.81, 99.72],
        'Recall (%)': [61.49, 87.84, 81.08, 83.78],
        'Precision (%)': [86, 6, 47, 36],
        'Frauds Caught': [91, 130, 120, 124],
        'Frauds Missed': [57, 18, 28, 24]
    }
    
    df_perf = pd.DataFrame(performance_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Recall Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#94a3b8', '#3b82f6', '#10b981', '#f59e0b']
        bars = ax.barh(df_perf['Model'], df_perf['Recall (%)'], color=colors)
        ax.set_xlabel('Recall (%)')
        ax.set_title('Fraud Detection Rate')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📊 Frauds Caught vs Missed")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_perf))
        width = 0.35
        ax.bar(x - width/2, df_perf['Frauds Caught'], width, label='Caught', color='#10b981')
        ax.bar(x + width/2, df_perf['Frauds Missed'], width, label='Missed', color='#ef4444')
        ax.set_xticks(x)
        ax.set_xticklabels(df_perf['Model'], rotation=15)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b;'><p>🔒 Credit Card Fraud Detection System</p></div>", unsafe_allow_html=True)