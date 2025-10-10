# app.py

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

st.title("💻 Full ML Project Workflow - Streamlit Tabs Version")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    st.subheader("First 5 rows of the dataset")
    st.dataframe(df.head())

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Training", "Evaluation", "Model Saving"])

    # ----------------------- TAB 1: EDA -----------------------
    with tab1:
        st.header("📊 Exploratory Data Analysis")
        
        st.subheader("Data Info")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Statistics")
        st.dataframe(df.describe())

        # Handle missing values
        df_clean = df.dropna().copy()
        st.write("✅ Dropped missing values. Dataset after cleaning:")
        st.dataframe(df_clean.head())

        # Encode categorical variables
        le = LabelEncoder()
        for col in df_clean.select_dtypes(include='object').columns:
            if col != 'file':
                df_clean[col] = le.fit_transform(df_clean[col])
        st.write("✅ Categorical columns encoded.")

        target_col = st.text_input("Enter the target column name:", key="eda_target")
        st.session_state['target_col'] = target_col  # store in session_state

        if target_col:
            if target_col not in df_clean.columns:
                st.error(f"Column '{target_col}' not found in dataset!")
            else:
                st.subheader(f"Distribution of {target_col}")
                sns.countplot(x=target_col, data=df_clean)
                st.pyplot()

                st.subheader("Correlation Heatmap")
                plt.figure(figsize=(12,8))
                numeric_cols = df_clean.select_dtypes(include=np.number).columns
                sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm')
                st.pyplot()

                st.subheader("Boxplots for numeric columns")
                for col in numeric_cols:
                    plt.figure()
                    sns.boxplot(y=df_clean[col])
                    plt.title(f"Boxplot of {col}")
                    st.pyplot()
        else:
            st.info("Please enter the target column name to visualize distributions and correlations.")

    # ----------------------- TAB 2: Training -----------------------
    with tab2:
        st.header("🛠 Training Models")
        
        if 'df_clean' not in locals():
            st.warning("Please complete EDA first.")
        else:
            target_col = st.session_state.get('target_col', None)
            if not target_col or target_col not in df_clean.columns:
                st.info("Enter a valid target column in the EDA tab first.")
            else:
                X = df_clean.drop([target_col, 'file'], axis=1, errors='ignore')
                y = df_clean[target_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                st.success("✅ Data split into training and testing sets.")

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                models = {
                    'RandomForest': RandomForestClassifier(random_state=42),
                    'SVM': SVC(probability=True, random_state=42),
                    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
                }

                for name, model in models.items():
                    if name == "RandomForest":
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train_scaled, y_train)
                    st.write(f"✅ {name} trained successfully")

    # ----------------------- TAB 3: Evaluation -----------------------
    with tab3:
        st.header("📈 Model Evaluation")
        if 'models' not in locals():
            st.warning("Please train the models first in the Training tab.")
        else:
            target_col = st.session_state.get('target_col', None)
            if not target_col or target_col not in df_clean.columns:
                st.info("Enter a valid target column in the EDA tab first.")
            else:
                X = df_clean.drop([target_col, 'file'], axis=1, errors='ignore')
                y = df_clean[target_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                results = []
                for name, model in models.items():
                    if name == "RandomForest":
                        y_pred = model.predict(X_test)
                    else:
                        y_pred = model.predict(X_test_scaled)

                    acc = accuracy_score(y_test, y_pred)
                    st.write(f"**{name} Accuracy:** {acc:.4f}")
                    st.text(classification_report(y_test, y_pred))

                    # Confusion matrix
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {name}')
                    st.pyplot()

                    # ROC Curve (binary only)
                    if len(y.unique()) == 2:
                        if name == "RandomForest":
                            y_prob = model.predict_proba(X_test)[:,1]
                        else:
                            y_prob = model.predict_proba(X_test_scaled)[:,1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        plt.plot(fpr, tpr, label=f'{name} ROC')
                        plt.plot([0,1], [0,1], 'k--')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve - {name}')
                        plt.legend()
                        st.pyplot()

                    results.append([name, acc])

                st.subheader("Feature Importance - RandomForest")
                rf_model = models['RandomForest']
                importances = rf_model.feature_importances_
                plt.figure(figsize=(10,6))
                plt.bar(X.columns, importances)
                plt.xticks(rotation=90)
                plt.title('Feature Importance - RandomForest')
                st.pyplot()

                st.subheader("Model Comparison")
                results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
                st.dataframe(results_df)

                st.subheader("Cross-validation (5-fold)")
                for name, model in models.items():
                    if name == "RandomForest":
                        scores = cross_val_score(model, X, y, cv=5)
                    else:
                        scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
                    st.write(f"{name} CV Mean Accuracy: {scores.mean():.4f}")

    # ----------------------- TAB 4: Model Saving -----------------------
    with tab4:
        st.header("💾 Save Best Model")
        target_col = st.session_state.get('target_col', None)
        if not target_col or target_col not in df_clean.columns:
            st.info("Enter a valid target column in the EDA tab first.")
        else:
            X = df_clean.drop([target_col, 'file'], axis=1, errors='ignore')
            y = df_clean[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            st.subheader("Hyperparameter tuning - RandomForest")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            st.write("Best RF Parameters:", grid.best_params_)
            st.write("Best RF Score:", grid.best_score_)

            best_model = grid.best_estimator_
            joblib.dump(best_model, 'final_model.pkl')
            st.success("✅ Best RandomForest model saved as 'final_model.pkl'")





