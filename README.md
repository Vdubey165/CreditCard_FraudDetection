# 💳 Credit Card Fraud Detection System

A machine learning web application built with Streamlit for detecting fraudulent credit card transactions using various ML models.

## 🎯 Project Overview

This application uses machine learning to detect fraudulent credit card transactions in real-time. The models were trained on a highly imbalanced dataset with only 0.173% fraudulent transactions, using advanced techniques like SMOTE to handle class imbalance.

## ✨ Features

- **Multiple ML Models**: Logistic Regression with SMOTE, Random Forest, and XGBoost
- **Single Transaction Prediction**: Check individual transactions in real-time
- **Batch Prediction**: Upload CSV files for bulk analysis
- **Model Performance Dashboard**: Compare model metrics and visualizations
- **High Recall Rate**: 87.84% recall to minimize missed fraudulent transactions
- **Interactive UI**: User-friendly Streamlit interface with visualizations

## 📊 Model Performance

| Model | Accuracy | Recall | Precision | Frauds Caught |
|-------|----------|--------|-----------|---------------|
| LR + SMOTE | 97.73% | **87.84%** | 6% | 130/148 |
| Random Forest | **99.81%** | 81.08% | **47%** | 120/148 |
| XGBoost | 99.72% | 83.78% | 36% | 124/148 |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fraud-detection-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have the trained models in the `models/` directory:
   - `lr_fraud_detector.pkl` (Logistic Regression model)
   - `rf_fraud_detector.pkl` (Random Forest model)
   - `scaler.pkl` (StandardScaler for preprocessing)

## 💻 Usage

### Running Locally

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Home Page**: View project overview and model performance summary
2. **Single Prediction**: 
   - Enter transaction details manually or generate a random sample
   - Select your preferred model
   - Get instant fraud prediction with probability scores
3. **Batch Prediction**:
   - Upload a CSV file with transaction data
   - Get predictions for all transactions
   - Download results as CSV
4. **Model Performance**: Compare different models and their metrics

## 📁 Project Structure

```
fraud-detection-app/
├── app.py                 # Main Streamlit application
├── models/                # Trained ML models
│   ├── lr_fraud_detector.pkl
│   ├── rf_fraud_detector.pkl
│   └── scaler.pkl
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Model Training

The models were trained using:
- **Dataset**: Credit Card Fraud Detection dataset (284,807 transactions)
- **Techniques**: SMOTE for handling class imbalance
- **Features**: 28 PCA-transformed features (V1-V28) + Time + Amount
- **Target**: Binary classification (0: Legitimate, 1: Fraud)

## 📈 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

**Important**: Make sure your `models/` directory contains the trained model files before deployment.

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization
- **imbalanced-learn**: SMOTE implementation

## 📝 Dataset Information

- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.173%)
- **Features**: 28 anonymized features (V1-V28) + Time + Amount
- **Source**: Credit card transactions by European cardholders (September 2013)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset provided by the Machine Learning Group - ULB
- Built with Streamlit
- ML models trained using scikit-learn

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational purposes. Always validate predictions with domain experts before using in production.
