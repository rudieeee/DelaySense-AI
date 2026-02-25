# DelaySense-AI
Amazon Supply Chain Intelligence Delay Predictor

**High accuracy** multi-class delivery delay predictor (On-Time/At Risk/Delayed)

## 🎯 Features
- DataCo Supply Chain dataset (180k orders)
- Advanced feature engineering (Haversine distance, risk scores)
- Ensemble ML (Logistic Regression + Decision Tree + Random Forest)
- SMOTE imbalance handling
- Streamlit dashboard
- SHAP explainability

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python preprocess.py
python train.py
streamlit run app.py
