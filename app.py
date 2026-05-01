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

st.set_page_config(page_title="Telecom Churn Predictor", layout="wide", page_icon="📞")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #111111 0%, #1f1f1f 100%);
    border-left: 5px solid #ef4444;
    padding: 1.4rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.75rem;
    color: white;
}
.hero h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }

.section-hdr {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-color, #6b7280);
    opacity: 0.65;
    margin: 0 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(128,128,128,0.25);
}

/* Prediction result cards */
.result-card { border-radius: 14px; padding: 1.4rem 1.6rem; }
.result-low    { background: #f9fafb; border: 1.5px solid #d1d5db; color: #111111; }
.result-medium { background: #fff1f2; border: 1.5px solid #fca5a5; color: #7f1d1d; }
.result-high   { background: #ef4444; border: 1.5px solid #dc2626; color: #ffffff; }
.result-icon   { font-size: 1.6rem; margin-bottom: 0.35rem; line-height: 1; }
.result-label  { font-size: 1.15rem; font-weight: 700; margin: 0; }
.result-sub    { font-size: 0.8rem; opacity: 0.75; margin: 0.3rem 0 0 0; }

/* Probability bar */
.prob-bar-wrap {
    background: #f3f4f6;
    border-radius: 9999px;
    height: 10px;
    overflow: hidden;
    margin: 0.4rem 0 0.1rem 0;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #555555 0%, #ef4444 100%);
}

/* KPI cards */
.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-val  { font-size: 1.4rem; font-weight: 700; color: #111111; margin: 0; line-height: 1.2; }
.kpi-name { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin: 0.3rem 0 0 0; }
.kpi-card.accent { border-top: 3px solid #ef4444; }

/* About page cards */
.about-card {
    border-left: 4px solid #ef4444;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    border-radius: 0 12px 12px 0;
    background: rgba(128,128,128,0.05);
}
.about-card h3 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #ef4444;
    margin: 0 0 0.6rem 0;
}
.about-card p, .about-card li {
    font-size: 0.92rem;
    line-height: 1.65;
    margin: 0.25rem 0;
}
.about-card ul { padding-left: 1.25rem; margin: 0.4rem 0 0 0; }
.stat-row { display: flex; gap: 1.5rem; margin-top: 0.75rem; flex-wrap: wrap; }
.stat-item { text-align: center; }
.stat-num  { font-size: 1.6rem; font-weight: 700; color: #f9fafb; line-height: 1; }
.stat-lbl  { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; }

/* Hide streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Sidebar base */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222222;
}
[data-testid="stSidebar"] section { background: #111111 !important; }

/* Light text for all sidebar content */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #9ca3af !important; }

/* Hide the default radio circle indicator */
[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] {
    display: none !important;
}

/* Nav item base */
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    display: flex !important;
    align-items: center;
    padding: 0.55rem 0.85rem !important;
    border-radius: 8px !important;
    border-left: 3px solid transparent !important;
    margin-bottom: 2px;
    transition: background 0.15s;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    background: rgba(255,255,255,0.05) !important;
    color: #f9fafb !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) {
    background: rgba(239,68,68,0.1) !important;
    border-left: 3px solid #ef4444 !important;
    color: #f9fafb !important;
}

/* Brand */
.sidebar-brand {
    font-size: 0.88rem;
    font-weight: 700;
    color: #f9fafb !important;
    letter-spacing: 0.01em;
    margin: 0 0 1.25rem 0.1rem;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global style ────────────────────────────────────────────────────
_C_BLACK = "#111111"
_C_RED   = "#ef4444"
_C_GRAY  = "#6b7280"

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linestyle'    : '--',
    'axes.labelcolor'   : _C_GRAY,
    'axes.titleweight'  : 'bold',
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 9,
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'xtick.color'       : _C_GRAY,
    'ytick.color'       : _C_GRAY,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : '#fafafa',
})


# ── Model loading ──────────────────────────────────────────────────────────────
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


# ── Preprocessing ──────────────────────────────────────────────────────────────
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


# ── Diagnostics (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_diagnostics(_model, threshold):
    df = pd.read_csv('data/preprocessed_churn.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_probs = _model.predict_proba(X_test)[:, 1]
    y_pred  = (y_probs >= threshold).astype(int)

    metrics = {
        'roc_auc'  : roc_auc_score(y_test, y_probs),
        'auc_pr'   : average_precision_score(y_test, y_probs),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f2'       : fbeta_score(y_test, y_pred, beta=2),
    }
    return y_test, y_probs, y_pred, metrics


# ── Hero + Sidebar ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📞 Telecom Churn Predictor</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="sidebar-brand">📞 Telecom Churn Predictor</p>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Customer Predictor", "Model Performance", "About"],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CUSTOMER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Customer Predictor":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-hdr">Demographics</p>', unsafe_allow_html=True)
        gender     = st.selectbox("Gender",          ["Female", "Male"])
        senior     = st.selectbox("Senior Citizen",  ["No", "Yes"])
        partner    = st.selectbox("Partner",         ["No", "Yes"])
        dependents = st.selectbox("Dependents",      ["No", "Yes"])

    with col2:
        st.markdown('<p class="section-hdr">Services</p>', unsafe_allow_html=True)
        phone          = st.selectbox("Phone Service",     ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
        internet       = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
        online_sec     = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_backup  = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_prot    = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support   = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        stream_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        stream_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    with col3:
        st.markdown('<p class="section-hdr">Account</p>', unsafe_allow_html=True)
        contract    = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
        paperless   = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment     = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check",
        ])
        tenure          = st.number_input("Tenure (months)",     min_value=0,   max_value=120, value=1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges   = st.number_input("Total Charges ($)",   min_value=0.0, value=50.0)

    st.write("")
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
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.metric("Churn Probability", f"{prob:.1%}")
                bar_w = f"{prob * 100:.1f}%"
                st.markdown(f"""
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{bar_w};"></div>
                </div>
                <p style="font-size:0.75rem;color:#9ca3af;margin:0.2rem 0 0 0;">
                    Decision threshold: {best_threshold:.1%}
                </p>
                """, unsafe_allow_html=True)

            with res_col2:
                if prob < best_threshold * 0.5:
                    css, icon, label, msg = (
                        "result-low", "🟢", "Low Risk",
                        "This customer shows minimal signs of churning.",
                    )
                elif prob < best_threshold:
                    css, icon, label, msg = (
                        "result-medium", "🟡", "Medium Risk",
                        "Some risk factors are present. Consider a proactive outreach.",
                    )
                else:
                    css, icon, label, msg = (
                        "result-high", "🔴", "High Risk",
                        "This customer is likely to churn. Immediate action is recommended.",
                    )

                st.markdown(f"""
                <div class="result-card {css}">
                    <div class="result-icon">{icon}</div>
                    <p class="result-label">{label}</p>
                    <p class="result-sub">{msg}</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Model Performance":
    st.markdown('<p class="section-hdr">Model Diagnostics</p>', unsafe_allow_html=True)

    if "diagnostics_loaded" not in st.session_state:
        st.session_state.diagnostics_loaded = False

    if not st.session_state.diagnostics_loaded:
        st.info("Diagnostics are computed on demand to keep the initial load fast.")
        if st.button("Load Diagnostics", type="primary"):
            st.session_state.diagnostics_loaded = True
            st.rerun()

    if st.session_state.diagnostics_loaded:
        try:
            with st.spinner("Computing metrics..."):
                y_test, y_probs, y_pred, metrics = load_diagnostics(model, best_threshold)

            kpis = [
                ("Threshold",  f"{best_threshold:.4f}", True),
                ("ROC-AUC",    f"{metrics['roc_auc']:.4f}",   False),
                ("AUC-PR",     f"{metrics['auc_pr']:.4f}",    False),
                ("Precision",  f"{metrics['precision']:.4f}", False),
                ("Recall",     f"{metrics['recall']:.4f}",    False),
                ("F2-Score",   f"{metrics['f2']:.4f}",        False),
            ]

            cols = st.columns(6)
            for col, (name, val, accent) in zip(cols, kpis):
                accent_cls = "accent" if accent else ""
                col.markdown(f"""
                <div class="kpi-card {accent_cls}">
                    <p class="kpi-val">{val}</p>
                    <p class="kpi-name">{name}</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            fig.patch.set_facecolor('white')

            # Confusion matrix
            cm      = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            annot   = np.array([
                [f"{v:,}\n({p:.1%})" for v, p in zip(r, rn)]
                for r, rn in zip(cm, cm_norm)
            ])
            axes[0, 0].grid(False)
            sns.heatmap(
                cm_norm, annot=annot, fmt='', cmap='Reds',
                ax=axes[0, 0], linewidths=2, linecolor='white',
                cbar_kws={'shrink': 0.72, 'label': 'Row-normalised rate'},
                annot_kws={'size': 11, 'weight': 'bold'},
            )
            axes[0, 0].set_title(f'Confusion Matrix  (threshold = {best_threshold:.3f})', pad=12)
            axes[0, 0].set_xlabel('Predicted Label')
            axes[0, 0].set_ylabel('True Label')
            axes[0, 0].set_xticklabels(['Stayed', 'Churned'], fontsize=9)
            axes[0, 0].set_yticklabels(['Stayed', 'Churned'], fontsize=9, rotation=0)

            # ROC curve
            fpr, tpr, roc_thr = roc_curve(y_test, y_probs)
            axes[0, 1].fill_between(fpr, tpr, alpha=0.10, color=_C_BLACK)
            axes[0, 1].plot(fpr, tpr, color=_C_BLACK, lw=2,
                            label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
            axes[0, 1].plot([0, 1], [0, 1], '--', color='#d1d5db', lw=1.2, label='Random baseline')
            op_idx = np.argmin(np.abs(roc_thr - best_threshold))
            axes[0, 1].scatter(
                fpr[op_idx], tpr[op_idx], color=_C_RED, zorder=5, s=90,
                edgecolors='white', linewidths=1.5,
                label=f'Operating point (t={best_threshold:.2f})',
            )
            axes[0, 1].set_title('ROC Curve', pad=12)
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_xlim([-0.02, 1.02])
            axes[0, 1].set_ylim([-0.02, 1.02])
            axes[0, 1].legend(fontsize=8, framealpha=0.8)

            # Precision-Recall curve
            precisions, recalls, pr_thr = precision_recall_curve(y_test, y_probs)
            prevalence = float(y_test.mean())
            axes[1, 0].fill_between(recalls, precisions, alpha=0.10, color=_C_BLACK)
            axes[1, 0].plot(recalls, precisions, color=_C_BLACK, lw=2,
                            label=f'AUC-PR = {metrics["auc_pr"]:.4f}')
            axes[1, 0].axhline(prevalence, color='#d1d5db', linestyle='--', lw=1.2,
                               label=f'Baseline (prevalence = {prevalence:.2f})')
            pr_idx = np.argmin(np.abs(pr_thr - best_threshold))
            axes[1, 0].scatter(
                recalls[pr_idx], precisions[pr_idx], color=_C_RED, zorder=5, s=90,
                edgecolors='white', linewidths=1.5,
                label=f'Operating point (t={best_threshold:.2f})',
            )
            axes[1, 0].set_title('Precision-Recall Curve', pad=12)
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_xlim([-0.02, 1.02])
            axes[1, 0].set_ylim([-0.02, 1.02])
            axes[1, 0].legend(fontsize=8, framealpha=0.8)

            # Feature importance
            importance = model.feature_importances_
            indices    = np.argsort(importance)[-10:]
            feat_vals  = importance[indices]
            feat_names = [feature_columns[i] for i in indices]

            norm   = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
            colors = [plt.cm.Reds(0.35 + 0.55 * v) for v in norm]

            bars = axes[1, 1].barh(
                range(len(indices)), feat_vals, color=colors, edgecolor='none', height=0.62,
            )
            for bar, val in zip(bars, feat_vals):
                axes[1, 1].text(
                    val + feat_vals.max() * 0.012, bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', fontsize=8, color=_C_GRAY,
                )
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels(feat_names, fontsize=8.5)
            axes[1, 1].set_title('Top 10 Feature Importances', pad=12)
            axes[1, 1].set_xlabel('Split Count')
            axes[1, 1].grid(axis='x', alpha=0.25, linestyle='--')
            axes[1, 1].grid(axis='y', visible=False)
            axes[1, 1].spines['left'].set_visible(False)

            plt.tight_layout(pad=2.5)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Failed to generate diagnostics: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
if page == "About":
    st.markdown('<p class="section-hdr">How This Works</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div class="about-card">
            <h3>The Dataset</h3>
            <p>This project uses the IBM Telco Customer Churn dataset, a widely used benchmark in
            customer retention research. It covers a fictional telecommunications company and
            contains one row per customer with information about the services they subscribed to,
            their account details, and whether they left the company within the last month.</p>
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-num">7,043</div>
                    <div class="stat-lbl">Customers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-num">19</div>
                    <div class="stat-lbl">Input Features</div>
                </div>
                <div class="stat-item">
                    <div class="stat-num">26%</div>
                    <div class="stat-lbl">Churn Rate</div>
                </div>
            </div>
        </div>

        <div class="about-card">
            <h3>Feature Engineering</h3>
            <p>Raw data requires several cleaning and encoding steps before a model can use it:</p>
            <ul>
                <li><b>Binary columns</b> (Yes/No) are mapped to 1 and 0.</li>
                <li>Values like "No internet service" and "No phone service" are normalised to "No"
                since they carry the same meaning as a plain "No" for the relevant feature.</li>
                <li><b>Numerical features</b> (tenure, monthly charges, total charges) are scaled
                to the 0-1 range using MinMaxScaler so that no single large number dominates the
                others during training.</li>
                <li><b>Categorical features</b> with more than two categories (Internet Service,
                Contract type, Payment Method) are one-hot encoded into separate binary columns.</li>
            </ul>
        </div>

        <div class="about-card">
            <h3>The Model</h3>
            <p>A gradient boosted decision tree classifier is used. This family of models is a
            strong default choice for structured tabular data because it handles mixed feature
            types natively, is robust to outliers and missing values, and consistently achieves
            top results on real-world classification benchmarks without heavy preprocessing.</p>
            <p>To account for the class imbalance (roughly 3 non-churners for every churner),
            the model is trained with <b>is_unbalance=True</b>, which automatically adjusts the
            internal sample weights so the minority class (churners) is not drowned out.</p>
            <p>The data is split 80/20 into training and test sets, stratified so the churn ratio
            is preserved in both halves.</p>
        </div>

        <div class="about-card">
            <h3>Threshold Selection</h3>
            <p>By default, a classifier flags a customer as a churner when the predicted
            probability exceeds 0.50. That default works well for balanced datasets but performs
            poorly here. Raising a false alarm (flagging a loyal customer) costs a small retention
            offer. Missing an actual churner costs the full lifetime value of that customer.</p>
            <p>To find the right balance, the threshold is chosen by maximising the
            <b>F2-score</b> on the training set. F2 is a variant of the F-score that weights
            recall twice as heavily as precision, directly encoding the business preference for
            catching churners over avoiding false alarms. The optimal threshold found is
            approximately <b>0.40</b>, well below the naive 0.50.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-card">
            <h3>Reading the Prediction</h3>
            <p>The model outputs a probability between 0 and 1. That probability is then compared
            against the decision threshold to produce a risk label:</p>
            <ul>
                <li><b>Low Risk</b> - probability is less than half the threshold. The customer
                shows no strong indicators of leaving.</li>
                <li><b>Medium Risk</b> - probability is between half the threshold and the
                threshold itself. Some warning signs are present and a light-touch outreach may
                be worthwhile.</li>
                <li><b>High Risk</b> - probability exceeds the threshold. The model is confident
                enough to recommend an active retention effort.</li>
            </ul>
        </div>

        <div class="about-card">
            <h3>Key Metrics</h3>
            <ul>
                <li><b>Recall (80%+):</b> Of all customers who actually churned, the model
                catches more than 8 in 10. This is the primary target metric.</li>
                <li><b>Precision (~49%):</b> Roughly half of the customers flagged as high risk
                genuinely do churn. Given the class imbalance, this is a strong result.</li>
                <li><b>ROC-AUC (0.83):</b> Measures the model's ability to rank a churner above
                a non-churner across all possible thresholds. A score above 0.80 indicates the
                model has learned real signal from the data.</li>
                <li><b>AUC-PR (0.64):</b> The area under the Precision-Recall curve. For
                imbalanced datasets this is a more honest measure than ROC-AUC, and 0.64
                comfortably outperforms a naive baseline of 0.26.</li>
            </ul>
        </div>

        <div class="about-card">
            <h3>Limitations</h3>
            <ul>
                <li>The model was trained on a single fictional telecom dataset and may not
                generalise to other markets or operators without retraining.</li>
                <li>Feature importance reflects how often a feature is used to split data, not
                its true causal effect on churn.</li>
                <li>The threshold was optimised on the training set. In production it should be
                re-evaluated periodically as customer behaviour shifts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
