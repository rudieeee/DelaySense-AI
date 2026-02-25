# Code Walkthrough: Where Everything Is
*Point to these sections when explaining your code*

---

## 📂 File: train.py (Model Training)

### Line 10-12: Importing the 3 Models
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
```
**Explain:** "These are the three models we learned in class that I'm using."

---

### Line 21-26: Creating Interaction Features
```python
X['distance_weekend'] = X['distance_km'] * X['is_weekend']
X['distance_weather'] = X['distance_km'] * X['weather_rain']
X['weekend_holiday'] = X['is_weekend'] * X['is_holiday_season']
X['distance_scheduled'] = X['distance_km'] * X['scheduled_days']
X['volume_distance'] = X['order_volume'] * X['distance_category']
```
**Explain:** "Feature engineering - creating interaction features. For example, distance matters more on weekends or in bad weather."

---

### Line 30: Train-Test Split (80/20)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```
**Explain:** "Splitting data 80% training, 20% testing. Stratify ensures balanced classes in both sets."

---

### Line 32-34: Feature Scaling
```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```
**Explain:** "Scaling features so they're on the same scale. Important for Logistic Regression."

---

### Line 36-37: SMOTE for Class Balance
```python
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
```
**Explain:** "SMOTE handles class imbalance by creating synthetic samples of minority classes."

---

### Line 39-68: Defining the 3 Models with Hyperparameters

#### Logistic Regression (Line 40-45)
```python
'LogisticRegression': LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight='balanced',
    solver='lbfgs',
    C=1.0  # Regularization strength
)
```
**Explain:** 
- "Statistical model using weighted features"
- "class_weight='balanced' handles class imbalance"
- "C=1.0 is regularization to prevent overfitting"

#### Decision Tree (Line 46-54)
```python
'DecisionTree': DecisionTreeClassifier(
    random_state=42,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    criterion='gini'
)
```
**Explain:**
- "Tree-based model creating if-then rules"
- "max_depth=20 prevents overfitting"
- "min_samples_split/leaf control tree complexity"

#### Random Forest (Line 55-65)
```python
'RandomForest': RandomForestClassifier(
    random_state=42,
    n_estimators=300,  # 300 trees!
    max_depth=20,
    class_weight='balanced',
    min_samples_split=10,
    max_features='sqrt',
    n_jobs=-1  # Use all CPU cores
)
```
**Explain:**
- "Builds 300 decision trees on random samples"
- "Each tree votes, majority wins"
- "n_jobs=-1 uses all CPU cores for speed"

---

### Line 70-81: Training Each Model
```python
for name, clf in models.items():
    print(f"Training {name}...")
    clf.fit(X_train_bal, y_train_bal)
    y_pred = clf.predict(X_test_s)
    acc = (y_test == y_pred).mean()
    print(f"{name} - Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred))
```
**Explain:** "Training each model on balanced training data, testing on unseen test data."

---

### Line 83-89: THE ENSEMBLE (Voting Classifier)
```python
ensemble = VotingClassifier([
    ('lr', models['LogisticRegression']),
    ('dt', models['DecisionTree']),
    ('rf', models['RandomForest'])
], voting='soft')  # ← IMPORTANT: soft voting averages probabilities
ensemble.fit(X_train_bal, y_train_bal)
```
**Explain:** 
- "Combines all 3 models"
- "voting='soft' means averaging probability predictions"
- "Each model contributes its probability estimate"

**Key Difference:**
- Hard voting: Each model votes for a class, majority wins
- Soft voting: Average the probabilities, pick highest

Example:
```
Model 1: [0.2, 0.7, 0.1] → predicts Class 1
Model 2: [0.3, 0.6, 0.1] → predicts Class 1  
Model 3: [0.4, 0.5, 0.1] → predicts Class 1

Soft voting average: [0.3, 0.6, 0.1] → Class 1 with 60% confidence
```

---

### Line 90-93: Ensemble Prediction
```python
y_pred_ens = ensemble.predict(X_test_s)
acc_ens = (y_test == y_pred_ens).mean()
print(f"Ensemble - Accuracy: {acc_ens*100:.2f}%")
```
**Explain:** "Testing ensemble on unseen data. Got 82.64% accuracy."

---

### Line 95-106: Results Summary
```python
print("FINAL RESULTS SUMMARY")
print(f"Test Set Size: {len(y_test)} samples")
print(f"\nModel Accuracies:")
for name in models.keys():
    print(f"  {name}: {accuracy_scores[name]*100:.2f}%")
print(f"  Ensemble: {acc_ens*100:.2f}% ✅")
```
**Explain:** "Comparing all models. Random Forest best at 82.88%, Ensemble at 82.64%."

---

### Line 126-143: Feature Importance (SHAP)
```python
explainer = shap.TreeExplainer(models['RandomForest'])
shap_values = explainer.shap_values(X_test_s)
joblib.dump(shap_values, 'feature_importance.joblib')
```
**Explain:** "SHAP explains which features are most important for predictions. Saved for use in Streamlit app."

---

### Line 147-150: Saving Models
```python
joblib.dump(ensemble, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns.tolist(), 'features.joblib')
```
**Explain:** "Saving trained models and scaler for use in production app."

---

## 📂 File: preprocess.py (Data Preparation)

### Line 68-70: Distance Category (NO Data Leakage!)
```python
df['distance_category'] = pd.cut(df['distance_km'], 
                                   bins=[0, 100, 500, 1000, 5000], 
                                   labels=[0,1,2,3]).astype(int)
```
**Explain:** "Binning distance into categories. This is available BEFORE delivery starts."

---

### Line 60-67: Temporal Features (NO Data Leakage!)
```python
df['day_of_week'] = df['Order_Date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['Order_Date'].dt.month
df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
```
**Explain:** "Extracting time-based features from ORDER date, not delivery date. Available BEFORE delivery."

---

### Line 73-75: Simulated Weather (NO Data Leakage!)
```python
np.random.seed(42)
df['weather_rain'] = np.random.binomial(1, 0.20, len(df))
df['peak_traffic'] = df['Order_Date'].dt.hour.fillna(12).isin([7,8,17,18,19]).astype(int)
```
**Explain:** "Simulating weather forecast (20% chance of rain) - available BEFORE delivery."

---

### Line 97-104: What We REMOVED (Data Leakage!)
```python
# ❌ REMOVED THESE - Only know AFTER delivery!
# - actual_days (only know after delivery completes)
# - delay_days (only know after comparing actual vs scheduled)
# - processing_time_days (only know after processing)
# - risk_score (calculated from actual delays)

# ✅ ONLY USING INFORMATION AVAILABLE BEFORE DELIVERY STARTS
return df[['scheduled_days', 'distance_km', 'order_volume', 
           'weather_rain', 'peak_traffic', 
           'day_of_week', 'is_weekend', 'month', 'is_holiday_season', 
           'distance_category']]  # + interaction features added in train.py
```
**Explain:** "These features were causing 100% accuracy because they're only known AFTER delivery. Removing them made prediction realistic and honest."

---

## 📂 File: app.py (Streamlit Web App)

### Line 232-252: Feature Calculation (Same as Training!)
```python
# Calculate temporal features
day_of_week = order_date.weekday()
is_weekend = 1 if day_of_week in [5, 6] else 0
month = order_date.month
is_holiday_season = 1 if month in [11, 12] else 0

# Calculate distance category
if distance <= 100:
    distance_category = 0
elif distance <= 500:
    distance_category = 1
elif distance <= 1000:
    distance_category = 2
else:
    distance_category = 3
```
**Explain:** "Calculating same features as in training. Must match exactly!"

---

### Line 254-269: Creating Same Interaction Features
```python
base_features = {...}

interaction_features = {
    'distance_weekend': distance * is_weekend,
    'distance_weather': distance * (1.0 if weather_rain else 0.0),
    'weekend_holiday': is_weekend * is_holiday_season,
    'distance_scheduled': distance * scheduled_days,
    'volume_distance': volume * distance_category
}

all_features = {**base_features, **interaction_features}
```
**Explain:** "Creating same 15 features (10 base + 5 interactions) as in training."

---

### Line 271-275: Making Prediction
```python
input_df = pd.DataFrame([all_features])[feature_names]
input_scaled = scaler.transform(input_df)
pred = model.predict(input_scaled)[0]
probs = model.predict_proba(input_scaled)[0]
```
**Explain:** 
- "Loading saved model and scaler"
- "Scaling input same way as training"
- "Getting prediction and probabilities"

---

## 🎯 KEY CODE LOCATIONS - QUICK REFERENCE

| What | File | Line | Code |
|------|------|------|------|
| Importing 3 models | train.py | 10-12 | `from sklearn...` |
| Interaction features | train.py | 21-26 | `X['distance_weekend'] = ...` |
| Train-test split | train.py | 30 | `train_test_split(..., test_size=0.2)` |
| SMOTE | train.py | 36-37 | `SMOTE(random_state=42)` |
| Logistic Regression params | train.py | 40-45 | `LogisticRegression(C=1.0, ...)` |
| Decision Tree params | train.py | 46-54 | `DecisionTreeClassifier(max_depth=20, ...)` |
| Random Forest params | train.py | 55-65 | `RandomForestClassifier(n_estimators=300, ...)` |
| **ENSEMBLE (voting)** | train.py | 83-89 | `VotingClassifier([...], voting='soft')` |
| Model training | train.py | 70-81 | `clf.fit(X_train_bal, y_train_bal)` |
| Saving models | train.py | 147-150 | `joblib.dump(ensemble, 'model.joblib')` |
| Temporal features | preprocess.py | 60-67 | `df['day_of_week'] = ...` |
| Distance category | preprocess.py | 68-70 | `pd.cut(df['distance_km'], ...)` |
| Removed leakage features | preprocess.py | 97-104 | Comments showing what was removed |
| Streamlit prediction | app.py | 271-275 | `model.predict(input_scaled)` |

---

## 💬 How to Reference Code in Presentation

**Example 1 - Explaining Ensemble:**
> "Here in train.py line 83-89, I create the VotingClassifier combining all three models using soft voting, which means it averages the probability predictions from each model."

**Example 2 - Explaining Data Leakage Fix:**
> "In preprocess.py line 97-104, you can see the features I removed - actual_days, delay_days, processing_time_days, and risk_score - because these are only known AFTER delivery completes. I only use information available BEFORE delivery starts."

**Example 3 - Explaining Interaction Features:**
> "In train.py line 21-26, I create interaction features like distance_weekend which multiplies distance by is_weekend. This captures that distance matters more on weekends when there's less delivery infrastructure."

**Example 4 - Explaining Random Forest:**
> "In train.py line 55-65, I configure Random Forest with 300 trees. Each tree sees a random sample of the data, and the final prediction is the majority vote of all 300 trees. n_jobs=-1 uses all CPU cores to train faster."

---

## 🎤 DEMONSTRATION FLOW

### 1. Show Streamlit App First
- "Let me demonstrate the working application..."
- Enter some values, show prediction
- Explain the features being used

### 2. Then Show Training Code
- "Now let me show how the model was trained..."
- Open train.py
- Point to the 3 models (line 39-68)
- Show ensemble creation (line 83-89)

### 3. Explain Data Leakage Fix
- "Here's what makes this project different..."
- Show preprocess.py
- Point to removed features (line 97-104)
- Explain temporal features (line 60-67)

### 4. Show Results
- "Here are the accuracy results..."
- Show terminal output or saved results
- Explain why 82.88% is good

---

Good luck! Point to specific lines when explaining - shows you understand the code! 🚀
