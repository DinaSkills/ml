import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('examples/full-Customer-Churn.csv')

# Binary encoding for 'Yes'/'No' columns
binary_columns = ['Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup', 'PaperlessBilling', 'Churn',
                  'PhoneService', 'MultipleLines']
for col in binary_columns:
    if col in data.columns:  # Ensure column exists
        data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encoding for categorical columns
data = pd.get_dummies(data,
                      columns=['gender', 'Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'StreamingTV',
                               'StreamingMovies', 'DeviceProtection'], drop_first=True)

# Convert numeric columns
numeric_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Handle missing values
data.fillna(0, inplace=True)

# Define features (X) and target (Y)
X = data.drop(columns=['customerID', 'Churn'], errors='ignore')
Y = data['Churn']

# Feature refinement based on importance
important_features = [
    'tenure', 'TotalCharges', 'MonthlyCharges',
    'Contract_Two year', 'InternetService_Fiber optic',
    'PaymentMethod_Electronic check', 'PaperlessBilling',
    'OnlineSecurity', 'Contract_One year'
]
X = X[important_features]  # Refine dataset to include only important features

# Verify class distribution
print("\nClass distribution before SMOTE:")
print(Y.value_counts())

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_train_scaled, Y_train)

# Verify class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(Y_resampled).value_counts())

# Initialize and train RandomForestClassifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_resampled, Y_resampled)

best_rf = grid_search.best_estimator_

# Make predictions with the best RandomForest model
rf_predictions = best_rf.predict(X_test_scaled)

# Evaluate the RandomForest model
rf_accuracy = accuracy_score(Y_test, rf_predictions)
rf_report = classification_report(Y_test, rf_predictions, zero_division=0)

print(f"\nRandomForest Accuracy: {rf_accuracy:.2f}")
print("RandomForest Classification Report:")
print(rf_report)

# Train and evaluate XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_resampled, Y_resampled)

xgb_predictions = xgb_model.predict(X_test_scaled)

# Evaluate the XGBoost model
xgb_accuracy = accuracy_score(Y_test, xgb_predictions)
xgb_report = classification_report(Y_test, xgb_predictions, zero_division=0)

print(f"\nXGBoost Accuracy: {xgb_accuracy:.2f}")
print("XGBoost Classification Report:")
print(xgb_report)

# Adjust the decision threshold for RandomForest predictions
rf_probabilities = best_rf.predict_proba(X_test_scaled)[:, 1]
threshold_predictions = (rf_probabilities > 0.4).astype(int)

# Evaluate the threshold-adjusted RandomForest model
threshold_accuracy = accuracy_score(Y_test, threshold_predictions)
threshold_report = classification_report(Y_test, threshold_predictions, zero_division=0)

print(f"\nThreshold-Adjusted RandomForest Accuracy: {threshold_accuracy:.2f}")
print("Threshold-Adjusted RandomForest Classification Report:")
print(threshold_report)

# Feature importance analysis for RandomForest
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': important_features, 'Importance': importances})
print("\nFeature Importance (RandomForest):")
print(feature_importance_df.sort_values(by='Importance', ascending=False))