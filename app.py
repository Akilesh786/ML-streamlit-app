import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
import joblib
from io import StringIO

# --- 1. PRO UI CONFIG ---
st.set_page_config(page_title="Purple ML Studio", page_icon="🔮", layout="wide")

# --- 2. THE "DEEP PURPLE" NEOMORPHIC CSS ---
st.markdown("""
    <style>
    /* Main App Background - Deep Night Purple */
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e, #16213e, #0f3460);
        color: #e94560;
    }

    /* Force all text to be visible (off-white/silver) */
    h1, h2, h3, h4, p, span, label {
        color: #e2e8f0 !important;
    }

    /* Sidebar - Solid Dark with Purple Border */
    section[data-testid="stSidebar"] {
        background-color: #0f3460 !important;
        border-right: 2px solid #6c63ff;
    }

    /* Glassmorphism Cards */
    div.stMetric, .stDataFrame, div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(108, 99, 255, 0.3);
    }

    /* Tabs Styling - Purple Neon */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(108, 99, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: white !important;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c63ff !important;
        border: 1px solid #6c63ff !important;
    }

    /* Custom Button - Neon Glow */
    div.stButton > button {
        background: linear-gradient(45deg, #6c63ff, #e94560) !important;
        color: white !important;
        border: none !important;
        font-weight: bold;
        transition: 0.3s ease all;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #6c63ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #6c63ff !important;'>🔮 STUDIO</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/8644/8644453.png", width=120)
    st.markdown("---")
    st.markdown("### ⚡ System Status")
    st.success("Core Engine: Active")
    st.info("Mode: Professional Analysis")

# --- 4. MAIN WORKFLOW ---
st.title("🚀 Full ML Project Workflow")
st.markdown("Transforming raw data into predictive intelligence.")

uploaded_file = st.file_uploader("📂 Upload Dataset (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # NEON DASHBOARD METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Samples", df.shape[0])
    with col2: st.metric("Dimensions", df.shape[1])
    with col3: st.metric("Missing", df.isna().sum().sum())
    with col4: st.metric("Categorical", len(df.select_dtypes(include='object').columns))

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 EDA", "🧠 Training", "📊 Evaluation", "📦 Export"])

    # ----------------------- TAB 1: EDA -----------------------
    with tab1:
        st.header("Exploratory Data Analysis")
        st.dataframe(df.head(), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Feature Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with c2:
            st.markdown("#### Data Structure")
            buffer = StringIO()
            df.info(buf=buffer)
            st.code(buffer.getvalue(), language="python")

        # Visuals
        df_clean = df.dropna().copy()
        le = LabelEncoder()
        for col in df_clean.select_dtypes(include='object').columns:
            if col != 'file': df_clean[col] = le.fit_transform(df_clean[col])

        target_col = st.text_input("🎯 Enter Target Column Name:", key="target_input")
        st.session_state['target_col'] = target_col

        if target_col and target_col in df_clean.columns:
            v1, v2 = st.columns(2)
            with v1:
                st.markdown(f"**{target_col} Distribution**")
                fig, ax = plt.subplots(facecolor='none')
                sns.countplot(x=target_col, data=df_clean, palette="magma", ax=ax)
                ax.set_title(f"Class Counts", color="white")
                ax.tick_params(colors='white')
                st.pyplot(fig)
            with v2:
                st.markdown("**Correlation Matrix**")
                fig, ax = plt.subplots(facecolor='none')
                sns.heatmap(df_clean.select_dtypes(include=np.number).corr(), cmap='Purples', ax=ax)
                ax.tick_params(colors='white')
                st.pyplot(fig)

    # ----------------------- TAB 2: Training -----------------------
    with tab2:
        st.header("Machine Learning Pipeline")
        if st.session_state.get('target_col') in df_clean.columns:
            target = st.session_state['target_col']
            X = df_clean.drop([target, 'file'], axis=1, errors='ignore')
            y = df_clean[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100),
                'SVM Classifier': SVC(probability=True),
                'Log Regression': LogisticRegression()
            }

            if st.button("🔥 Start Neural Engine"):
                with st.status("Training models...", expanded=True) as status:
                    for name, model in models.items():
                        st.write(f"Training {name}...")
                        model.fit(X_train if 'Forest' in name else X_train_s, y_train)
                    status.update(label="Training Sequence Complete!", state="complete")
                    st.session_state['trained_models'] = models
        else:
            st.info("Please set the target column in the EDA tab.")

    # ----------------------- TAB 3: Evaluation -----------------------
    with tab3:
        st.header("Performance Analytics")
        if 'trained_models' in st.session_state:
            for name, model in st.session_state['trained_models'].items():
                with st.expander(f"Analysis: {name}"):
                    y_pred = model.predict(X_test if 'Forest' in name else X_test_s)
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Model Accuracy", f"{acc:.4f}")
                    st.code(classification_report(y_test, y_pred))
                    
                    fig, ax = plt.subplots(figsize=(6,3), facecolor='none')
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='PuRd', ax=ax)
                    ax.tick_params(colors='white')
                    st.pyplot(fig)
        else:
            st.warning("No models found. Please train models in the 'Training' tab.")

    # ----------------------- TAB 4: Export -----------------------
    with tab4:
        st.header("Deployment Center")
        if 'trained_models' in st.session_state:
            st.success("Best model ready for deployment.")
            joblib.dump(st.session_state['trained_models']['Random Forest'], 'model.pkl')
            with open("model.pkl", "rb") as f:
                st.download_button("📥 Download Pickle Model", f, "trained_model.pkl")
        else:
            st.info("Train your models to unlock the export options.")

else:
    st.markdown("""
        <div style='text-align: center; padding: 50px; border: 2px dashed #6c63ff; border-radius: 20px;'>
            <h3>System Standby</h3>
            <p>Please upload an Excel dataset to initialize the AI modules.</p>
        </div>
    """, unsafe_allow_html=True)
