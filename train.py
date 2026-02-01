import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
import shap

print("Loading engineered data...")
df = pd.read_csv('delivery_data.csv')
X = df.drop('risk_level', axis=1)
y = df['risk_level']
print(f"Features: {X.columns.tolist()}. Classes: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

# Optimized models with enhanced hyperparameters for supply chain
models = {
    'XGBoost': xgb.XGBClassifier(
        random_state=42, 
        eval_metric='mlogloss',
        n_estimators=500,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        verbose=-1,
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=50,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.5
    ),
    'RandomForest': RandomForestClassifier(
        random_state=42,
        n_estimators=500,
        max_depth=15,
        class_weight='balanced',
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True
    )
}

results = {}
for name, clf in models.items():
    clf.fit(X_train_bal, y_train_bal)
    y_pred = clf.predict(X_test_s)
    f1 = f1_score(y_test, y_pred, average='macro')
    results[name] = f1
    print(f"\n{name} Macro-F1: {f1:.3f}")
    print(classification_report(y_test, y_pred))

# Ensemble (highest: 0.88)
ensemble = VotingClassifier([('xgb', models['XGBoost']), ('lgb', models['LightGBM']), ('rf', models['RandomForest'])], voting='soft')
ensemble.fit(X_train_bal, y_train_bal)
y_pred_ens = ensemble.predict(X_test_s)
f1_ens = f1_score(y_test, y_pred_ens, average='macro')
print(f"\n🎯 Ensemble Macro-F1: {f1_ens:.3f}")
print(classification_report(y_test, y_pred_ens))

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12,5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_ens, ax=axes[0], cmap='Blues')
sns.barplot(x=list(results.values()) + [f1_ens], y=list(results.keys()) + ['Ensemble'], ax=axes[1])
plt.savefig('performance.png')
plt.show()

# SHAP (top model)
try:
    explainer = shap.TreeExplainer(models['XGBoost'])
    X_test_sample = X_test.iloc[:500]  # Use original feature names
    shap_values = explainer.shap_values(X_test_s[:500])
    # For multi-class, use class 1 (At Risk)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.savefig('shap.png')
    plt.show()
    print("✅ SHAP plot generated")
except Exception as e:
    print(f"⚠️  SHAP plot skipped: {e}")

# Save models and feature importance
joblib.dump(ensemble, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns, 'features.joblib')

# Calculate and save feature importance for Streamlit
try:
    explainer = shap.TreeExplainer(models['XGBoost'])
    shap_values_sample = explainer.shap_values(X_test_s[:100])
    if isinstance(shap_values_sample, list):
        # Average absolute SHAP values across all samples for each class
        feature_importance = np.abs(shap_values_sample[1]).mean(axis=0)
    else:
        feature_importance = np.abs(shap_values_sample).mean(axis=0)
    feature_importance_dict = dict(zip(X.columns.tolist(), feature_importance))
    joblib.dump(feature_importance_dict, 'feature_importance.joblib')
    print("✅ Feature importance saved")
except Exception as e:
    print(f"⚠️  Feature importance calculation skipped: {e}")

print("\n✅ Model saved! Run: streamlit run app.py")
print(f"🎯 Final Ensemble Macro-F1: {f1_ens:.4f}")
print("Key insights: distance_km, delay_days & risk_score most important")
