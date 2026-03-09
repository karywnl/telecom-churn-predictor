import streamlit as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, average_precision_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")

# Ensure required libraries are imported for plotting
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_artifacts():
    return joblib.load('models/model.pkl')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    best_threshold = artifacts['threshold']
    feature_columns = artifacts['features']
except Exception as e:
    st.error(f"Failed to load model artifacts. Please ensure 'model.ipynb' has been executed. Error: {e}")
    st.stop()


def preprocess_input(input_data):
    """
    Preprocess the raw input dictionary exactly as done in pre-processing.ipynb.
    """
    df = pd.DataFrame([input_data])
    
    # TotalCharges handling
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isnull().iloc[0]:
        df['TotalCharges'] = 0.0 # Default fallback if empty
        
    # Standardize 'No internet service' and 'No phone service'
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)
    
    # Binary encoding
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
                     
    for col in yes_no_columns:
        if col in df.columns:
            df[col].replace({'Yes': 1, 'No': 0}, inplace=True)
            
    # Gender encoding
    if 'gender' in df.columns:
        df['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
        
    # One-hot encoding for categorical variables
    # We create a dataframe with all possible expected columns initialized to 0
    df2 = pd.DataFrame(columns=feature_columns)
    df2.loc[0] = 0
    
    # Map matched columns
    for col in df.columns:
        if col in df2.columns:
            df2[col] = df[col]
            
    # Map one-hot encoded variables
    if 'InternetService' in df.columns:
        val = df['InternetService'].iloc[0]
        col_name = f'InternetService_{val}'
        if col_name in df2.columns:
            df2[col_name] = 1
            
    if 'Contract' in df.columns:
        val = df['Contract'].iloc[0]
        col_name = f'Contract_{val}'
        if col_name in df2.columns:
            df2[col_name] = 1
            
    if 'PaymentMethod' in df.columns:
        val = df['PaymentMethod'].iloc[0]
        col_name = f'PaymentMethod_{val}'
        if col_name in df2.columns:
            df2[col_name] = 1
            
    # Scale numerical columns
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df2[cols_to_scale] = scaler.transform(df2[cols_to_scale])
    
    # Ensure correct column order
    df_final = df2[feature_columns]
    
    return df_final


# UI Setup
st.title("📞 Telecom Churn Predictor")

tab1, tab2 = st.tabs(["Customer Predictor", "Model Performance"])

with tab1:
    st.header("Assess Customer Churn Risk")
    st.write("Enter the customer's details below to predict their likelihood of churning.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        stream_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    with col3:
        st.subheader("Account")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)
        
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': stream_tv,
            'StreamingMovies': stream_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        try:
            processed_data = preprocess_input(input_data)
            
            # Predict
            prob = model.predict_proba(processed_data)[0][1]
            will_churn = prob >= best_threshold
            
            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Churn Probability", f"{prob:.1%}")
                
            with res_col2:
                if prob < best_threshold * 0.5:
                    risk = "Low Risk 🟢"
                    st.success(f"Prediction: **{risk}**")
                elif prob < best_threshold:
                    risk = "Medium Risk 🟡"
                    st.warning(f"Prediction: **{risk}**")
                else:
                    risk = "High Risk 🔴"
                    st.error(f"Prediction: **{risk}**")
                    
                st.caption(f"Based on tuned threshold constraint of {best_threshold:.1%}")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

with tab2:
    st.header("Model Performance & Diagnostics")
    
    @st.cache_data
    def load_evaluation_data():
        df = pd.read_csv('data/preprocessed_churn.csv')
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_test, y_test
        
    try:
        X_test, y_test = load_evaluation_data()
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= best_threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_probs)
        auc_pr = average_precision_score(y_test, y_probs)
        
        # Calculate derived threshold metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Classification Threshold", f"{best_threshold:.4f}")
        m_col2.metric("ROC-AUC", f"{roc_auc:.4f}")
        m_col3.metric("AUC-PR", f"{auc_pr:.4f}")
        
        m_col4, m_col5, m_col6 = st.columns(3)
        m_col4.metric("Precision", f"{precision:.4f}")
        m_col5.metric("Recall", f"{recall:.4f}")
        m_col6.metric("F2-Score", f"{f2:.4f}")
        
        st.divider()
        
        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix (Threshold = {best_threshold:.2f})')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        axes[0, 1].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        
        # 3. Precision-Recall Curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        axes[1, 0].plot(recalls, precisions, label=f'AUC-PR = {auc_pr:.4f}')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        
        # Plot the exact threshold on the graph
        valid_idx = np.argmin(np.abs(thresholds - best_threshold))
        axes[1, 0].scatter([recalls[valid_idx]], [precisions[valid_idx]], color='red', marker='o', s=100, label=f'Tuned Thresh ({best_threshold:.2f})')
        axes[1, 0].legend()
        
        # 4. Feature Importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[-10:] # Top 10 features
        
        axes[1, 1].barh(range(len(indices)), importance[indices], align='center')
        axes[1, 1].set_yticks(range(len(indices)))
        axes[1, 1].set_yticklabels([feature_columns[i] for i in indices])
        axes[1, 1].set_title('Top 10 Feature Importances (Split)')
        axes[1, 1].set_xlabel('LightGBM Feature Importance')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Failed to generate dashboard charts. Error: {e}")
