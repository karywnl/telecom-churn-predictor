import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, fbeta_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")


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
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()


def preprocess_input(input_data):
    df = pd.DataFrame([input_data])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)

    yes_no_columns = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    ]
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

    df2 = pd.DataFrame(columns=feature_columns)
    df2.loc[0] = 0

    for col in df.columns:
        if col in df2.columns:
            df2[col] = df[col]

    for col, prefix in [
        ('InternetService', 'InternetService_'),
        ('Contract', 'Contract_'),
        ('PaymentMethod', 'PaymentMethod_'),
    ]:
        if col in df.columns:
            col_name = f"{prefix}{df[col].iloc[0]}"
            if col_name in df2.columns:
                df2[col_name] = 1

    df2[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df2[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    return df2[feature_columns]


@st.cache_data
def load_diagnostics(_model, threshold):
    """Heavy computation for Tab 2 — cached after first run."""
    df = pd.read_csv('data/preprocessed_churn.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_probs = _model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_probs),
        'auc_pr': average_precision_score(y_test, y_probs),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f2': fbeta_score(y_test, y_pred, beta=2),
    }

    return y_test, y_probs, y_pred, metrics


# ── UI ────────────────────────────────────────────────────────────────────────

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
        payment = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check",
        ])
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
            'TotalCharges': total_charges,
        }
        try:
            processed_data = preprocess_input(input_data)
            prob = model.predict_proba(processed_data)[0][1]

            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric("Churn Probability", f"{prob:.1%}")

            with res_col2:
                if prob < best_threshold * 0.5:
                    st.success("Prediction: **Low Risk 🟢**")
                elif prob < best_threshold:
                    st.warning("Prediction: **Medium Risk 🟡**")
                else:
                    st.error("Prediction: **High Risk 🔴**")
                st.caption(f"Based on tuned threshold of {best_threshold:.1%}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

with tab2:
    st.header("Model Performance & Diagnostics")

    if "diagnostics_loaded" not in st.session_state:
        st.session_state.diagnostics_loaded = False

    if not st.session_state.diagnostics_loaded:
        st.info("Diagnostics are loaded on demand to keep the app fast.")
        if st.button("Load Diagnostics", type="primary"):
            st.session_state.diagnostics_loaded = True
            st.rerun()

    if st.session_state.diagnostics_loaded:
        try:
            with st.spinner("Computing metrics..."):
                y_test, y_probs, y_pred, metrics = load_diagnostics(model, best_threshold)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Classification Threshold", f"{best_threshold:.4f}")
            m_col2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            m_col3.metric("AUC-PR", f"{metrics['auc_pr']:.4f}")

            m_col4, m_col5, m_col6 = st.columns(3)
            m_col4.metric("Precision", f"{metrics['precision']:.4f}")
            m_col5.metric("Recall", f"{metrics['recall']:.4f}")
            m_col6.metric("F2-Score", f"{metrics['f2']:.4f}")

            st.divider()

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title(f'Confusion Matrix (Threshold = {best_threshold:.2f})')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')

            fpr, tpr, _ = roc_curve(y_test, y_probs)
            axes[0, 1].plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.4f}')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].legend()

            precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
            axes[1, 0].plot(recalls, precisions, label=f'AUC-PR = {metrics["auc_pr"]:.4f}')
            valid_idx = np.argmin(np.abs(thresholds - best_threshold))
            axes[1, 0].scatter(
                [recalls[valid_idx]], [precisions[valid_idx]],
                color='red', marker='o', s=100,
                label=f'Tuned Threshold ({best_threshold:.2f})',
            )
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()

            importance = model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(len(indices)), importance[indices], align='center')
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([feature_columns[i] for i in indices])
            axes[1, 1].set_title('Top 10 Feature Importances (Split)')
            axes[1, 1].set_xlabel('LightGBM Feature Importance')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Failed to generate diagnostics: {e}")
