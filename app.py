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

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="AI Model Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR GLASSMORPHISM & STYLING ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom Card Style */
    div.stButton > button:first-child {
        background-color: #4A90E2;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.5);
        padding: 10px;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103807.png", width=100)
    st.title("AutoML Studio")
    st.info("Upload your data and let the AI do the heavy lifting.")
    st.divider()
    st.markdown("### 🛠️ Settings")
    theme = st.selectbox("Color Palette", ["Cool Blue", "Emerald City", "Sunset"])
    st.write("Current Status: **Ready to process**")

# --- MAIN APP ---
st.title("🚀 Full ML Project Workflow")
st.markdown("#### Transform raw data into production-ready models in minutes.")

# Step 1: Upload dataset
with st.container():
    st.subheader("📁 Data Acquisition")
    uploaded_file = st.file_uploader("Drop your Excel file here", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Summary Metrics for a "Dashboard" feel
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Categorical", len(df.select_dtypes(include='object').columns))
    col4.metric("Numeric", len(df.select_dtypes(include=np.number).columns))

    st.divider()

    # Tabs with Emoji Icons
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "⚙️ Training", "📈 Evaluation", "💾 Deployment"])

    # ----------------------- TAB 1: EDA -----------------------
    with tab1:
        st.header("Exploratory Data Analysis")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.expander("🔍 Data Info & Types", expanded=True):
                buffer = StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
        
        with c2:
            with st.expander("🔢 Statistical Summary", expanded=True):
                st.dataframe(df.describe(), use_container_width=True)

        # Preprocessing Section
        st.markdown("---")
        st.subheader("🛠️ Quick Preprocessing")
        df_clean = df.dropna().copy()
        st.success(f"✅ Auto-cleaned: {df.shape[0] - df_clean.shape[0]} missing rows removed.")

        le = LabelEncoder()
        for col in df_clean.select_dtypes(include='object').columns:
            if col != 'file':
                df_clean[col] = le.fit_transform(df_clean[col])
        st.caption("Categorical columns automatically encoded using LabelEncoder.")

        # Visualization Section
        target_col = st.text_input("🎯 Enter the target column name (e.g., 'Target'):", key="eda_target")
        st.session_state['target_col'] = target_col

        if target_col:
            if target_col not in df_clean.columns:
                st.error(f"Column '{target_col}' not found!")
            else:
                viz1, viz2 = st.columns(2)
                with viz1:
                    st.subheader("Target Distribution")
                    fig, ax = plt.subplots()
                    sns.countplot(x=target_col, data=df_clean, palette="viridis", ax=ax)
                    st.pyplot(fig)

                with viz2:
                    st.subheader("Correlation Heatmap")
                    fig, ax = plt.subplots()
                    numeric_cols = df_clean.select_dtypes(include=np.number).columns
                    sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)

    # ----------------------- TAB 2: Training -----------------------
    with tab2:
        st.header("Model Training Center")
        
        if 'df_clean' not in locals():
            st.warning("Please complete EDA first.")
        else:
            target_col = st.session_state.get('target_col', None)
            if not target_col or target_col not in df_clean.columns:
                st.info("Enter a valid target column in the EDA tab.")
            else:
                with st.status("Training models...", expanded=True) as status:
                    X = df_clean.drop([target_col, 'file'], axis=1, errors='ignore')
                    y = df_clean[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    models = {
                        'RandomForest': RandomForestClassifier(random_state=42),
                        'SVM': SVC(probability=True, random_state=42),
                        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
                    }

                    for name, model in models.items():
                        st.write(f"Fitting {name}...")
                        if name == "RandomForest":
                            model.fit(X_train, y_train)
                        else:
                            model.fit(X_train_scaled, y_train)
                    status.update(label="All models trained successfully!", state="complete")

    # ----------------------- TAB 3: Evaluation -----------------------
    with tab3:
        st.header("Performance Analytics")
        if 'models' not in locals():
            st.warning("Please train the models first.")
        else:
            # Re-running the split logic (matching your original code)
            X = df_clean.drop([target_col, 'file'], axis=1, errors='ignore')
            y = df_clean[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            results = []
            for name, model in models.items():
                with st.container():
                    st.markdown(f"### Results for {name}")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    if name == "RandomForest":
                        y_pred = model.predict(X_test)
                    else:
                        y_pred = model.predict(X_test_scaled)

                    acc = accuracy_score(y_test, y_pred)
                    results.append([name, acc])

                    with res_col1:
                        st.metric("Accuracy", f"{acc:.2%}")
                        st.text("Classification Report:")
                        st.code(classification_report(y_test, y_pred))

                    with res_col2:
                        fig, ax = plt.subplots(figsize=(5,3))
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                st.divider()

    # ----------------------- TAB 4: Model Saving -----------------------
    with tab4:
        st.header("Finalize & Export")
        if 'df_clean' not in locals() or 'target_col' not in st.session_state:
            st.info("Complete previous steps first.")
        else:
            st.subheader("🏆 Auto-Tuning Best Model")
            with st.spinner("Running GridSearch..."):
                param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
                grid.fit(X_train, y_train)
                
                st.balloons()
                st.success("Tuning Complete!")
                
                c1, c2 = st.columns(2)
                c1.json(grid.best_params_)
                
                best_model = grid.best_estimator_
                joblib.dump(best_model, 'final_model.pkl')
                
                with open("final_model.pkl", "rb") as f:
                    st.download_button("📥 Download Pickle File", f, "trained_model.pkl")

else:
    # Empty State
    st.info("Please upload an Excel file to begin the analysis.")
