import joblib
import pandas as pd
from sklearn.metrics import classification_report
# Run after train.py
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
df = pd.read_csv('delivery_data.csv')
# ... (load X_test etc from train or re-split)
# Prints detailed report
print("Full evaluation ready.")
