import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def preprocess_data(data, important_features):
    """
    Preprocess the dataset for training
    """

    # Binary encoding for 'Yes'/'No' columns
    binary_columns = ['Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup', 'PaperlessBilling', 'Churn',
                      'PhoneService', 'MultipleLines']
    for col in binary_columns:
        if col in data.columns:  # Ensure column exists
            data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)

    data = pd.get_dummies(data,
                          columns=['gender', 'Contract', 'InternetService', 'PaymentMethod', 'TechSupport',
                                   'StreamingTV',
                                   'StreamingMovies', 'DeviceProtection'], drop_first=True)

    # Convert numeric columns
    numeric_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Handle missing values
    data.fillna(0, inplace=True)

    # Select important features
    X = data[important_features]
    Y = data['Churn']
    return X, Y


# Function to train and evaluate models
def train_models(X, Y):
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X_train_scaled, Y_train)

    # Initialize and train RandomForestClassifier with hyperparameter tuning

    # Make predictions with the best RandomForest model
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_resampled, Y_resampled)
    rf_predictions = rf_model.predict(X_test_scaled)
    rf_report = classification_report(Y_test, rf_predictions, zero_division=0, output_dict=True)

    # Initialize and train XGBoost model
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_resampled, Y_resampled)
    xgb_predictions = xgb_model.predict(X_test_scaled)
    # Evaluate the XGBoost model
    xgb_report = classification_report(Y_test, xgb_predictions, zero_division=0,  output_dict=True)

    return rf_report, xgb_report, rf_model.feature_importances_, xgb_model.feature_importances_,

# Streamlit App
st.title("Model Performance Comparison")
uploaded_file = st.file_uploader("Upload Dataset(CSV format)", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Sample:")
    st.write(data.head())

    important_features = [
        'tenure', 'TotalCharges', 'MonthlyCharges',
        'Contract_Two year', 'InternetService_Fiber optic',
        'PaymentMethod_Electronic check', 'PaperlessBilling',
        'OnlineSecurity', 'Contract_One year'
    ]

    X, Y = preprocess_data(data, important_features)
    rf_report, xgb_report, rf_feature_importances, xgb_feature_importances = train_models(X, Y)
    st.sidebar.header("Random Forest Classifier")
    model_option = st.sidebar.radio("Select Model", ["Random Forest", "XGBoost"])
    if model_option == "Random Forest":
        st.write("RandomForest Model")
        st.json(rf_report)
        st.subheader("Feature Importance (Random Forest)")
        feature_importance_df = pd.DataFrame({'Feature': important_features, 'Importance': rf_feature_importances})
        st.dataframe(feature_importance_df.sort_values(by='Importance', ascending=False))
    else:
        st.write("XGBoost Model")
        st.json(xgb_report)
        st.subheader("Feature Importance (XGBoost)")
        feature_importance_df = pd.DataFrame({'Feature': important_features, 'Importance': xgb_feature_importances})
        st.dataframe(feature_importance_df.sort_values(by='Importance', ascending=False))


else:
    st.write("Please upload a dataset to proceed.")