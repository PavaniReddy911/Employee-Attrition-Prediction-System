import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI-Powered Employee Attrition Prediction System")

uploaded_file = st.file_uploader(
    "Upload IBM HR Attrition Dataset (.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # =====================
        # Attrition Rate
        # =====================

        if "Attrition" in df.columns:

            attrition_rate = (
                (df["Attrition"] == "Yes").sum()
                / len(df)
            ) * 100

            st.metric(
                "Attrition Rate",
                f"{attrition_rate:.2f}%"
            )

        # =====================
        # Preprocessing
        # =====================

        df_ml = df.copy()

        cols_to_drop = [
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ]

        existing_cols = [
            col for col in cols_to_drop
            if col in df_ml.columns
        ]

        df_ml.drop(
            columns=existing_cols,
            inplace=True
        )

        # Encode all object columns
        categorical_cols = df_ml.select_dtypes(
            include=["object"]
        ).columns

        for col in categorical_cols:
            encoder = LabelEncoder()
            df_ml[col] = encoder.fit_transform(
                df_ml[col].astype(str)
            )

        # =====================
        # Features & Target
        # =====================

        X = df_ml.drop("Attrition", axis=1)
        y = df_ml["Attrition"]

        # Convert everything to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.fillna(0)

        # =====================
        # Scaling
        # =====================

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # =====================
        # Train Test Split
        # =====================

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        # =====================
        # Random Forest
        # =====================

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.subheader("Model Performance")

        st.success(
            f"Random Forest Accuracy: {accuracy:.4f}"
        )

        # =====================
        # Feature Importance
        # =====================

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        st.subheader("Top 10 Important Features")

        st.dataframe(
            importance_df.head(10),
            use_container_width=True
        )

        st.bar_chart(
            importance_df.head(10)
            .set_index("Feature")
        )

        # =====================
        # Sample Prediction
        # =====================

        st.subheader("Employee Risk Prediction")

        sample_employee = X.iloc[[0]]

        sample_scaled = scaler.transform(
            sample_employee
        )

        prediction = rf.predict(
            sample_scaled
        )[0]

        if prediction == 1:
            st.error(
                "High Attrition Risk"
            )
        else:
            st.success(
                "Low Attrition Risk"
            )

        st.subheader("Dataset Information")

        st.write(
            f"Rows: {df.shape[0]}"
        )

        st.write(
            f"Columns: {df.shape[1]}"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Please upload the IBM HR Attrition CSV file."
    )
