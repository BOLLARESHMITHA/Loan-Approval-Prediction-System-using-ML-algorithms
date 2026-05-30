import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark banking theme */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1524 !important;
    border-right: 1px solid #1e2d4a;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 50%, #071020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.03em;
}
.main-header p {
    color: #7a8fa8;
    font-size: 1rem;
    margin: 0;
    font-family: 'DM Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #0f1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'DM Mono', monospace;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #5a7290;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a5f;
}

/* Prediction result */
.pred-approved {
    background: linear-gradient(135deg, #0a2e1a, #0a1f0e);
    border: 1px solid #1a7a3a;
    border-left: 4px solid #00e676;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    font-size: 1.5rem;
    font-weight: 800;
    color: #00e676;
}
.pred-rejected {
    background: linear-gradient(135deg, #2e0a0a, #1f0a0a);
    border: 1px solid #7a1a1a;
    border-left: 4px solid #ff4444;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    font-size: 1.5rem;
    font-weight: 800;
    color: #ff5555;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0060cc, #003f99);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0070ee, #0050bb);
    transform: translateY(-1px);
}

/* Info/success/warning boxes */
.stAlert {
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}

/* Slider */
.stSlider > div { font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────
IMPORTANT_FEATURES = [
    'Age', 'InterestRate', 'Income', 'MonthsEmployed',
    'LoanAmount', 'HasCoSigner_Yes', 'EmploymentType_Unemployed',
    'HasDependents_Yes', 'CreditScore'
]

@st.cache_data
def load_and_preprocess(uploaded_file):
    data = pd.read_csv(uploaded_file)
    raw = data.copy()
    data.drop(columns=['LoanID'], inplace=True, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    return raw, data

@st.cache_resource
def train_models(data):
    # ensure all needed dummy cols exist
    for col in ['HasCoSigner_Yes', 'EmploymentType_Unemployed', 'HasDependents_Yes']:
        if col not in data.columns:
            data[col] = 0

    X = data[IMPORTANT_FEATURES]
    y = data['Default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)

    # Logistic Regression
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_sm, y_train_sm)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sm, y_train_sm)

    # KMeans
    X_scaled_all = scaler.transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled_all)

    return {
        "scaler": scaler,
        "lr": lr,
        "rf": rf,
        "km": km,
        "clusters": clusters,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "X": X,
        "y": y,
    }

def model_metrics(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    return {
        "acc":       round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc":   round(roc_auc_score(y_test, y_prob) * 100, 2),
        "cm":        confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }


# ──────────────────────────────────────────────
# PLOT HELPERS (dark-themed)
# ──────────────────────────────────────────────
DARK_BG   = "#0a0e1a"
CARD_BG   = "#0f1a2e"
BORDER    = "#1e3a5f"
ACCENT    = "#00d4ff"
GREEN     = "#00e676"
RED       = "#ff4444"
TEXT      = "#c8d8e8"

def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    return fig, ax

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = dark_fig(5, 4)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        linewidths=0.5, linecolor=BORDER,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.tick_params(colors=TEXT)
    fig.tight_layout()
    return fig

def plot_roc(y_test, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = dark_fig(5, 4)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], '--', color='#3a5a7a', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return fig

def plot_feature_importance(model, features):
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = dark_fig(6, 4)
    colors = [ACCENT if i >= len(imp)-3 else '#1e4a7a' for i in range(len(imp))]
    imp.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
    ax.set_title("Feature Importance", fontsize=12, fontweight='bold')
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig

def plot_clusters(X, clusters):
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    X_pca = pca.fit_transform(scaler.fit_transform(X))
    fig, ax = dark_fig(6, 5)
    palette = [ACCENT, GREEN, "#ff9800"]
    for i, c in enumerate(palette):
        mask = clusters == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=c, alpha=0.55, s=12, label=f"Cluster {i+1}", edgecolors='none')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Customer Segments (PCA)", fontsize=12, fontweight='bold')
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return fig

def plot_elbow(X):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        wcss.append(km.inertia_)
    fig, ax = dark_fig(6, 4)
    ax.plot(range(1, 11), wcss, marker='o', color=ACCENT, lw=2, markersize=7,
            markerfacecolor=GREEN, markeredgewidth=0)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method", fontsize=12, fontweight='bold')
    ax.grid(True, color=BORDER, alpha=0.5, linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig

def plot_class_dist(y):
    counts = y.value_counts()
    fig, ax = dark_fig(4, 3)
    bars = ax.bar(["Not Default", "Default"], counts.values,
                  color=[GREEN, RED], edgecolor='none', width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + counts.max()*0.01,
                f'{b.get_height():,}', ha='center', va='bottom',
                color=TEXT, fontsize=10, fontweight='bold')
    ax.set_title("Class Distribution", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count")
    ax.grid(axis='y', color=BORDER, alpha=0.5, linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 Loan Approval System</h1>
  <p>ML-powered risk assessment · Logistic Regression · Random Forest · K-Means</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR — DATA UPLOAD
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload Loan_default.csv", type=["csv"])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#5a7290; font-family:"DM Mono",monospace;'>
    Expected columns:<br>
    LoanID, Age, Income, LoanAmount,<br>
    CreditScore, MonthsEmployed,<br>
    InterestRate, EmploymentType,<br>
    HasCoSigner, HasDependents,<br>
    Default
    </div>
    """, unsafe_allow_html=True)

if uploaded is None:
    st.info("👆 Upload **Loan_default.csv** (from the Kaggle dataset) in the sidebar to get started.")
    st.markdown("""
    **Dataset:** [Loan Default — Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

    Once uploaded, this app will:
    - Train Logistic Regression & Random Forest classifiers
    - Run K-Means customer segmentation
    - Let you make live predictions for new applicants
    """)
    st.stop()


# ──────────────────────────────────────────────
# LOAD & TRAIN
# ──────────────────────────────────────────────
with st.spinner("🔄 Loading data and training models…"):
    raw, data = load_and_preprocess(uploaded)
    trained   = train_models(data)

lr_m = model_metrics(trained["lr"], trained["X_test_scaled"], trained["y_test"])
rf_m = model_metrics(trained["rf"], trained["X_test_scaled"], trained["y_test"])

# Sidebar — model selector
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    active_model = st.radio("Active prediction model",
                            ["Logistic Regression", "Random Forest"], index=1)
    st.markdown("---")
    st.markdown(f"**Rows:** `{len(raw):,}`")
    st.markdown(f"**Features used:** `{len(IMPORTANT_FEATURES)}`")
    st.markdown(f"**Default rate:** `{raw['Default'].mean()*100:.1f}%`")


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🤖 Logistic Regression",
    "🌲 Random Forest",
    "🔵 Clustering",
    "🔮 Predict"
])


# ── TAB 1: OVERVIEW ──────────────────────────
with tab1:
    st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="val">{len(raw):,}</div><div class="lbl">Total Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val">{raw["Default"].sum():,}</div><div class="lbl">Defaults</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="val">{raw["Default"].mean()*100:.1f}%</div><div class="lbl">Default Rate</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="val">{raw.shape[1]}</div><div class="lbl">Columns</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_class_dist(raw['Default']))

    with col_r:
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(raw.head(20), use_container_width=True, height=320)

    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(raw.describe().round(2), use_container_width=True)


# ── TAB 2: LOGISTIC REGRESSION ────────────────
with tab2:
    st.markdown('<div class="section-title">Logistic Regression · Loan Approval / Rejection</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, "Accuracy",   f"{lr_m['acc']}%"),
        (c2, "Precision",  f"{lr_m['precision']}%"),
        (c3, "ROC-AUC",    f"{lr_m['roc_auc']}%"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.pyplot(plot_confusion_matrix(lr_m["cm"], "Confusion Matrix — LR"))
    with col_r:
        st.pyplot(plot_roc(trained["y_test"], lr_m["y_prob"], "ROC Curve — LR"))

    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(lr_m["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)


# ── TAB 3: RANDOM FOREST ──────────────────────
with tab3:
    st.markdown('<div class="section-title">Random Forest · Risk Prediction</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, "Accuracy",   f"{rf_m['acc']}%"),
        (c2, "Precision",  f"{rf_m['precision']}%"),
        (c3, "ROC-AUC",    f"{rf_m['roc_auc']}%"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.pyplot(plot_confusion_matrix(rf_m["cm"], "Confusion Matrix — RF"))
    with col_r:
        st.pyplot(plot_roc(trained["y_test"], rf_m["y_prob"], "ROC Curve — RF"))

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    st.pyplot(plot_feature_importance(trained["rf"], IMPORTANT_FEATURES))

    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(rf_m["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)


# ── TAB 4: CLUSTERING ─────────────────────────
with tab4:
    st.markdown('<div class="section-title">K-Means · Customer Segmentation</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Elbow Method**")
        st.pyplot(plot_elbow(trained["X"]))
    with col_r:
        st.markdown("**Cluster Visualisation (PCA)**")
        st.pyplot(plot_clusters(trained["X"], trained["clusters"]))

    st.markdown('<div class="section-title">Cluster Profiles</div>', unsafe_allow_html=True)
    profile = trained["X"].copy()
    profile["Cluster"] = trained["clusters"] + 1
    st.dataframe(
        profile.groupby("Cluster").mean().round(2),
        use_container_width=True
    )


# ── TAB 5: PREDICT ────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Live Applicant Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"**Active model:** `{active_model}` *(change in sidebar)*")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        age              = st.slider("Age", 18, 75, 35)
        income           = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        credit_score     = st.slider("Credit Score", 300, 850, 650)

    with col_b:
        loan_amount      = st.number_input("Loan Amount ($)", 1000, 500000, 25000, step=500)
        interest_rate    = st.slider("Interest Rate (%)", 1.0, 30.0, 8.5, step=0.1)
        months_employed  = st.slider("Months Employed", 0, 360, 48)

    with col_c:
        has_cosigner     = st.selectbox("Has Co-Signer?",  ["No", "Yes"])
        employment_type  = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        has_dependents   = st.selectbox("Has Dependents?", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Run Prediction", use_container_width=True):
        input_data = {
            "Age":                        age,
            "InterestRate":               interest_rate,
            "Income":                     income,
            "MonthsEmployed":             months_employed,
            "LoanAmount":                 loan_amount,
            "HasCoSigner_Yes":            1 if has_cosigner == "Yes" else 0,
            "EmploymentType_Unemployed":  1 if employment_type == "Unemployed" else 0,
            "HasDependents_Yes":          1 if has_dependents == "Yes" else 0,
            "CreditScore":                credit_score,
        }

        df_input = pd.DataFrame([input_data])[IMPORTANT_FEATURES]
        scaled   = trained["scaler"].transform(df_input)

        model     = trained["lr"] if active_model == "Logistic Regression" else trained["rf"]
        pred      = model.predict(scaled)[0]
        prob      = model.predict_proba(scaled)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 0:
            st.markdown(
                f'<div class="pred-approved">✅ Loan APPROVED &nbsp;·&nbsp; '
                f'Confidence: {prob[0]*100:.1f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="pred-rejected">❌ Loan REJECTED &nbsp;·&nbsp; '
                f'Default Risk: {prob[1]*100:.1f}%</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown(f"**P(No Default):** `{prob[0]*100:.2f}%`")
        with pc2:
            st.markdown(f"**P(Default):** `{prob[1]*100:.2f}%`")

        # Probability bar
        fig, ax = dark_fig(6, 1.2)
        ax.barh(["Risk"], [prob[1]], color=RED, height=0.5, edgecolor='none')
        ax.barh(["Risk"], [prob[0]], left=[prob[1]], color=GREEN, height=0.5, edgecolor='none')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Default Risk Distribution", fontsize=10)
        ax.axvline(0.5, color='white', lw=0.8, linestyle='--', alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("📋 Input Summary"):
            st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: "Value"}), use_container_width=True)
