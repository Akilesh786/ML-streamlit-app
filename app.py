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

# --- 1. SET PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="ML Studio Pro", page_icon="🧪", layout="wide")

# --- 2. THE "FORCE READABLE" CSS ---
st.markdown("""
    <style>
    /* Force a clean background that works for both themes */
    .stApp {
        background: linear-gradient(to bottom right, #eff6ff, #dbeafe);
    }
    
    /* Make Title and Headers pop with deep blue contrast */
    h1, h2, h3, .stMarkdown p {
        color: #1e3a8a !important; 
        font-family: 'Inter', sans-serif;
    }

    /* Style the Sidebar to be distinct */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e2e8f0;
    }

    /* Style the Tabs for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 12px;
        padding: 5px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Improve File Uploader Visibility */
    section[data-testid="stFileUploadDropzone"] {
        background-color: white !important;
        border: 2px dashed #3b82f6 !important;
        border-radius: 15px;
    }

    /* Clean Card containers */
    div[data-testid="stVerticalBlock"] > div.element-container {
        background: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103807.png", width=80)
    st.title("AutoML Studio")
    st.markdown("---")
    st.write("🔧 **System Status:** Online")
    st.info("Upload an Excel file to unlock the dashboard.")

# --- 4. MAIN APP LOGIC ---
st.title("🚀 ML Project Workflow")
st.markdown("##### Upload your data to begin automated feature engineering and model training.")

uploaded_file = st.file_uploader("Drop your dataset here", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Dashboard Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", df.shape[0])
    m2.metric("Features", df.shape[1])
    m3.metric("Missing Values", df.isna().sum().sum())

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "⚙️ Train Models", "📈 Performance", "💾 Export"])

    # ----------------------- TAB 1: EDA -----------------------
    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Column Metadata**")
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with col_b:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(df.describe())

        df_clean = df.dropna().copy()
        le = LabelEncoder()
        for col in df_clean.select_dtypes(include='object').columns:
            if col != 'file':
                df_clean[col] = le.fit_transform(df_clean[col])

        target_col = st.text_input("🎯 Identify Target Column:", key="eda_target")
        st.session_state['target_col'] = target_col

        if target_col and target_col in df_clean.columns:
            v1, v2 = st.columns(2)
            with v1:
                fig, ax = plt.subplots()
                sns.countplot(x=target_col, data=df_clean, palette="Blues")
                st.pyplot(fig)
            with v2:
                fig, ax = plt.subplots()
                numeric_cols = df_clean.select_dtypes(include=np.number).columns
                sns.heatmap(df_clean[numeric_cols].corr(), cmap='RdBu', ax=ax)
                st.pyplot(fig)

    # ----------------------- TAB 2: Training -----------------------
    with tab2:
        if 'df_clean' in locals() and st.session_state.get('target_col'):
            st.subheader("Model Pipeline")
            target = st.session_state['target_col']
            
            X = df_clean.drop([target, 'file'], axis=1, errors='ignore')
            y = df_clean[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models = {
                'RandomForest': RandomForestClassifier(),
                'SVM': SVC(probability=True),
                'Logistic': LogisticRegression()
            }

            with st.status("Training Engines...", expanded=True):
                for name, model in models.items():
                    st.write(f"Processing {name}...")
                    model.fit(X_train if name == 'RandomForest' else X_train_s, y_train)
                st.success("Training Complete!")
        else:
            st.warning("Please specify a Target Column in the Analysis tab.")

    # ----------------------- TAB 3: Evaluation -----------------------
    with tab3:
        if 'models' in locals():
            for name, model in models.items():
                with st.expander(f"Report: {name}", expanded=(name=='RandomForest')):
                    y_p = model.predict(X_test if name == 'RandomForest' else X_test_s)
                    acc = accuracy_score(y_test, y_p)
                    st.metric(f"{name} Accuracy", f"{acc:.2%}")
                    st.code(classification_report(y_test, y_p))

    # ----------------------- TAB 4: Model Saving -----------------------
    with tab4:
        if 'models' in locals():
            st.subheader("Model Serialization")
            joblib.dump(models['RandomForest'], 'model.pkl')
            with open("model.pkl", "rb") as f:
                st.download_button("💾 Download .PKL File", f, "best_model.pkl")

else:
    st.empty()
    st.info("Waiting for data upload...")
