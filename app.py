import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.markdown("""
<style>

/* Background */
.stApp{
background: linear-gradient(
135deg,
#020617,
#0f172a,
#1e293b
);
overflow:hidden;
}

/* AIRPLANE */

.airplane{
position:fixed;
top:80px;
left:-200px;
font-size:60px;
z-index:9999;
animation:fly 18s linear infinite;
}

@keyframes fly{
0%{
left:-200px;
transform:translateY(0px);
}
50%{
transform:translateY(-40px);
}
100%{
left:110%;
transform:translateY(0px);
}
}

/* EMPLOYEE RIGHT */

.employee1{
position:fixed;
bottom:40px;
left:-100px;
font-size:50px;
animation:walkright 15s linear infinite;
}

@keyframes walkright{
0%{
left:-100px;
}
100%{
left:110%;
}
}

/* EMPLOYEE LEFT */

.employee2{
position:fixed;
bottom:120px;
right:-100px;
font-size:50px;
animation:walkleft 18s linear infinite;
}

@keyframes walkleft{
0%{
right:-100px;
}
100%{
right:110%;
}
}

/* AI GLOW */

.glow{
font-size:80px;
text-align:center;
animation:pulse 2s infinite;
}

@keyframes pulse{
0%{
transform:scale(1);
}
50%{
transform:scale(1.2);
}
100%{
transform:scale(1);
}
}

</style>

<div class="airplane">✈️</div>

<div class="employee1">👨‍💼</div>

<div class="employee2">👩‍💼</div>

<div class="glow">🤖</div>

""", unsafe_allow_html=True)

st.set_page_config(
    page_title="HR Command Center",
    page_icon="🚀",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#020617,
#0f172a,
#1e293b
);
}

.block-container{
padding-top:2rem;
}

.title{
font-size:50px;
font-weight:700;
color:white;
text-align:center;
}

.card{
background:rgba(255,255,255,0.08);
padding:25px;
border-radius:20px;
backdrop-filter:blur(10px);
border:1px solid rgba(255,255,255,0.2);
text-align:center;
}

.metric{
font-size:35px;
font-weight:bold;
color:#00E5FF;
}

.metric-title{
font-size:18px;
color:white;
}

.insight{
background:#0f172a;
padding:20px;
border-radius:15px;
border-left:5px solid cyan;
color:white;
}

section[data-testid="stSidebar"]{
background:#0f172a;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
"""
<div class='title'>
🚀 HR COMMAND CENTER
</div>
""",
unsafe_allow_html=True
)

st.write("")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"Select",
[
"Dashboard",
"EDA",
"Prediction",
"Insights"
]
)

uploaded_file = st.sidebar.file_uploader(
"Upload Dataset",
type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # =============================
    # PREPROCESSING
    # =============================

    df_ml = df.copy()

    drop_cols = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    for col in drop_cols:
        if col in df_ml.columns:
            df_ml.drop(col, axis=1, inplace=True)

    cat_cols = df_ml.select_dtypes(
        include="object"
    ).columns

    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(
            df_ml[col].astype(str)
        )

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

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )

    # ===================================
    # DASHBOARD
    # ===================================

    if page == "Dashboard":

        total_emp = len(df)

        attrition_rate = (
            (df["Attrition"]=="Yes").sum()
            / len(df)
        )*100

        avg_income = int(
            df["MonthlyIncome"].mean()
        )

        avg_age = int(
            df["Age"].mean()
        )

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.markdown(
            f"""
            <div class='card'>
            <div class='metric'>{total_emp}</div>
            <div class='metric-title'>
            Employees
            </div>
            </div>
            """,
            unsafe_allow_html=True
            )

        with c2:
            st.markdown(
            f"""
            <div class='card'>
            <div class='metric'>
            {attrition_rate:.2f}%
            </div>
            <div class='metric-title'>
            Attrition
            </div>
            </div>
            """,
            unsafe_allow_html=True
            )

        with c3:
            st.markdown(
            f"""
            <div class='card'>
            <div class='metric'>
            ₹{avg_income}
            </div>
            <div class='metric-title'>
            Avg Salary
            </div>
            </div>
            """,
            unsafe_allow_html=True
            )

        with c4:
            st.markdown(
            f"""
            <div class='card'>
            <div class='metric'>
            {avg_age}
            </div>
            <div class='metric-title'>
            Avg Age
            </div>
            </div>
            """,
            unsafe_allow_html=True
            )

        st.write("")

        st.subheader("🤖 Model Accuracy")

        st.progress(int(accuracy*100))

        st.success(
            f"Accuracy : {accuracy*100:.2f}%"
        )

    # ===================================
    # EDA
    # ===================================

    elif page == "EDA":

        st.subheader("Department Wise Attrition")

        fig = px.histogram(
            df,
            x="Department",
            color="Attrition",
            barmode="group"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Gender Wise Attrition")

        fig2 = px.pie(
            df,
            names="Gender"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.subheader("Monthly Income")

        fig3 = px.box(
            df,
            x="Attrition",
            y="MonthlyIncome"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ===================================
    # PREDICTION
    # ===================================

    elif page == "Prediction":

        st.subheader(
            "Employee Attrition Prediction"
        )

        sample = X.iloc[[0]]

        result = rf.predict(
            scaler.transform(sample)
        )[0]

        risk = rf.predict_proba(
            scaler.transform(sample)
        )[0][1]

        st.metric(
            "Risk Score",
            f"{risk*100:.2f}%"
        )

        st.progress(
            int(risk*100)
        )

        if result == 1:
            st.error(
                "⚠ High Attrition Risk"
            )
        else:
            st.success(
                "✅ Low Attrition Risk"
            )

    # ===================================
    # AI INSIGHTS
    # ===================================

    elif page == "Insights":

        importance = pd.DataFrame(
            {
            "Feature":X.columns,
            "Importance":
            rf.feature_importances_
            }
        )

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        st.subheader(
            "Top Drivers of Attrition"
        )

        fig = px.bar(
            importance.head(10),
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
        """
        <div class='insight'>

        🔥 Key Findings

        • Overtime strongly influences attrition

        • Low salary employees are vulnerable

        • Sales department has higher turnover

        • Employees with low satisfaction show higher risk

        • HR should focus on retention programs

        </div>
        """,
        unsafe_allow_html=True
        )

else:

    st.info(
    "⬅ Upload IBM HR Dataset from Sidebar"
    )
