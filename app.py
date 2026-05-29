import streamlit as st
import pandas as pd
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

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        attrition_rate = (
            df["Attrition"].value_counts().get("Yes", 0)
            / len(df)
        ) * 100

        st.metric(
            "Attrition Rate",
            f"{attrition_rate:.2f}%"
        )

    st.divider()

    st.subheader("⚙️ Data Preprocessing")

    df_ml = df.copy()

    drop_cols = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    existing_cols = [
        col for col in drop_cols
        if col in df_ml.columns
    ]

    df_ml.drop(
        columns=existing_cols,
        inplace=True
    )

    le = LabelEncoder()

    for col in df_ml.columns:
        if df_ml[col].dtype == "object":
            df_ml[col] = le.fit_transform(df_ml[col])

    X = df_ml.drop("Attrition", axis=1)
    y = df_ml["Attrition"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    st.success("Preprocessing Completed Successfully")

    st.divider()

    st.subheader("🤖 Random Forest Model Training")

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

    st.metric(
        "Model Accuracy",
        f"{accuracy:.4f}"
    )

    st.divider()

    st.subheader("🔥 Top 10 Important Features")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance_df.head(10),
        use_container_width=True
    )

    st.bar_chart(
        importance_df.head(10).set_index("Feature")
    )

    st.divider()

    st.subheader("🚨 Employee Risk Prediction")

    sample_employee = X.iloc[[0]]

    sample_scaled = scaler.transform(
        sample_employee
    )

    risk_prediction = rf.predict(
        sample_scaled
    )[0]

    if risk_prediction == 1:
        st.error("High Attrition Risk")
    else:
        st.success("Low Attrition Risk")

    st.divider()

    st.subheader("📋 Model Summary")

    st.write(
        """
        ✅ Dataset Uploaded Successfully
        
        ✅ Data Preprocessing Completed
        
        ✅ Random Forest Model Trained
        
        ✅ Feature Importance Generated
        
        ✅ Employee Attrition Risk Predicted
        """
    )

else:

    st.info(
        "Upload the IBM HR Analytics Employee Attrition Dataset to start."
    )
