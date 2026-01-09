# =====================================================
# REQUIRED PACKAGES
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import streamlit.components.v1 as components

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4



# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Study Track AI", layout="wide")

# =====================================================
# SESSION STATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "model" not in st.session_state:
    st.session_state.model = None

# =====================================================
# HELPER: AUTO COLUMN DETECTION
# =====================================================
def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key in col:
                return col
    return None

# =====================================================
# GLOBAL CSS
# =====================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color:#0f0f0f; }
[data-testid="stSidebar"] { background-color:#111111; }

h1,h2,h3,h4,h5,h6,p,label {
    color:#ff9f1a !important;
    font-weight:bold !important;
}

/* ---------- LOGIN BOX ---------- */
.login-box {
    background:#1c1c1c;
    padding:40px;
    width:420px;
    border-radius:16px;
    box-shadow:0 0 25px rgba(255,159,26,0.6);
}
.login-title {
    font-size:28px;
    text-align:center;
    margin-bottom:25px;
}

.stTextInput input {
    background:#1c1c1c;
    color:white;
    border:1px solid #ff9f1a;
    border-radius:8px;
}

.stButton > button {
    background:#ff9f1a;
    color:black;
    font-weight:bold;
    border-radius:8px;
}

/* ---------- RIGHT ORANGE PANEL ---------- */
.login-info {
    background: linear-gradient(135deg, #ff9f1a, #ff7a00);
    padding: 42px;
    border-radius: 18px;
    box-shadow: 0 25px 55px rgba(0,0,0,0.4);
    animation: slideInRight 1s ease-out forwards;
}

@keyframes slideInRight {
    from { transform: translateX(60px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.login-info h2 {
    color:#000;
    font-size:30px;
    margin-bottom:15px;
}

.login-info p {
    color:#000;
    font-size:15px;
}

/* ---------- FEATURE ROW ---------- */
.feature {
    display:flex;
    align-items:center;
    background:rgba(255,255,255,0.45);
    padding:12px 16px;
    border-radius:12px;
    margin-bottom:12px;
    font-size:15px;
    font-weight:700;
    color:#000;
    transition:all 0.3s ease;
}

.feature:hover {
    transform: translateX(6px) scale(1.03);
    background:rgba(255,255,255,0.7);
    box-shadow:0 10px 25px rgba(0,0,0,0.3);
}

.feature span {
    font-size:20px;
    margin-right:12px;
 }
           
 /* ---------- HOME HOVER PANELS ---------- */
.hover-container {
    display: flex;
    gap: 30px;
    margin-top: 35px;
}

.hover-panel {
    flex: 1;
    background: linear-gradient(135deg, #1c1c1c, #262626);
    border-left: 6px solid #ff9f1a;
    padding: 30px;
    border-radius: 16px;
    transition: all 0.35s ease;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
}

.hover-panel:hover {
    transform: translateY(-12px) scale(1.03);
    box-shadow: 0 25px 45px rgba(255,159,26,0.55);
}

.hover-title {
    font-size: 26px;
    font-weight: 800;
    color: #ff9f1a;
    margin-bottom: 14px;
    text-transform: uppercase;
}

.hover-text {
    font-size: 16px;
    line-height: 1.6;
    color: #f5f5f5;
}
/* ---------- PAGE HEADER ---------- */
.page-header {
    display: flex;
    align-items: center;
    gap: 18px;
    background: #1c1c1c;
    padding: 18px 24px;
    border-radius: 14px;
    margin-bottom: 30px;
    box-shadow: 0 0 18px rgba(255,159,26,0.45);
    transition: all 0.35s ease;
}

.page-header:hover {
    transform: translateX(6px);
    box-shadow: 0 0 30px rgba(255,159,26,0.8);
}

/* Orange vertical bar */
.page-bar {
    width: 6px;
    height: 42px;
    background: #ff9f1a;
    border-radius: 6px;
}

/* Header text */
.page-title {
    font-size: 26px;
    font-weight: 800;
    color: #ff9f1a;
    letter-spacing: 1px;
}

/* Black horizontal separator */
.page-line {
    width: 100%;
    height: 2px;
    background: linear-gradient(to right, #ff9f1a, #000);
    margin-top: 10px;
    border-radius: 2px;
}
 
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
def page_header(title):
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-bar"></div>
            <div class="page-title">{title}</div>
        </div>
        <div class="page-line"></div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# RULE-BASED TEST SCORE CALCULATION (0–10)
# -------------------------------------------------
def calculate_test_score(row):
    """
    Rule-based scoring to avoid ML collapse
    """
    score = (
        row["Study Hours"] * 1.2 +
        row["Sleep Hours"] * 0.4 -
        row["Play Hours"] * 0.8 -
        row["Other Hours"] * 0.2
    )

    # Clamp between 0 and 10
    return round(max(1, min(10, score)), 2)

# =====================================================
# LOGIN PAGE
# =====================================================
def login_page():
    page_header("STUDY TRACK AI – STUDENT STUDY HABIT RECOMMENDER")

    left, right = st.columns([1.1, 1.4])

    # LEFT LOGIN FORM
    with left:
        st.markdown("""
        <div class="login-box">
            <div class="login-title">🔐 STUDY TRACK AI LOGIN</div>
        """, unsafe_allow_html=True)

        full_name = st.text_input("FULL NAME")
        email = st.text_input("EMAIL")
        username = st.text_input("USERNAME")
        password = st.text_input("PASSWORD", type="password")

        if st.button("LOGIN", use_container_width=True):
            if full_name and email and username and password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("PLEASE FILL ALL FIELDS")

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT INFO PANEL
    with right:
     components.html(
        """
        <style>
            .panel {
                background: linear-gradient(135deg, #ff9f1a, #ff7a00);
                padding: 40px;
                border-radius: 18px;
                box-shadow: 0 25px 55px rgba(0,0,0,0.4);
                animation: slideIn 1s ease-out forwards;
                font-family: Arial, sans-serif;
            }

            @keyframes slideIn {
                from {
                    transform: translateX(60px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            .panel h2 {
                color: black;
                font-size: 30px;
                margin-bottom: 16px;
            }

            .panel p {
                color: black;
                font-size: 15px;
                margin-bottom: 18px;
            }

            .feature {
                display: flex;
                align-items: center;
                background: rgba(255,255,255,0.5);
                padding: 12px 16px;
                border-radius: 12px;
                margin-bottom: 12px;
                font-size: 15px;
                font-weight: bold;
                color: black;
                transition: all 0.3s ease;
            }

            .feature:hover {
                transform: translateX(6px);
                background: rgba(255,255,255,0.8);
            }

            .icon {
                font-size: 20px;
                margin-right: 12px;
            }
        </style>
    

        <div class="panel">
            <h2>STUDY TRACK AI</h2>

            <p>
                AI-powered system to analyze student study habits
                and predict academic performance.
            </p>

            <div class="feature">
                <span class="icon">📈</span>
                AI-driven test score prediction
            </div>

            <div class="feature">
                <span class="icon">🧠</span>
                Study, sleep & play habit analysis
            </div>

            <div class="feature">
                <span class="icon">🎯</span>
                Personalized recommendations
            </div>

            <div class="feature">
                <span class="icon">📂</span>
                CSV & Excel dataset support
            </div>

            <p><strong>Infosys Springboard Capstone Project</strong></p>
        </div>
        """,
        height=520
    )


# =====================================================
# DASHBOARD
# =====================================================
def dashboard():
    st.sidebar.title("📘 STUDY TRACK AI")

    menu = st.sidebar.radio(
        "NAVIGATION",
        [
            "HOME",
            "MODEL TRAINING",
            "SINGLE PREDICTION",
            "BULK PREDICTION",
            "STUDENT RECOMMENDATION"
        ]
    )

    if menu == "HOME":
        home()

    elif menu == "MODEL TRAINING":
        model_training()

    elif menu == "SINGLE PREDICTION":
        single_prediction()

    elif menu == "BULK PREDICTION":
        batch_prediction()

    elif menu == "STUDENT RECOMMENDATION":
        recommendation()


# -------------------------------------------------
# HOME
# -------------------------------------------------
def home():
    page_header("STUDY TRACK AI DASHBOARD")
    st.write("")

    # HERO IMAGE
    st.image("study_track_banner.png", use_container_width=True)

    # MISSION & VISION (FORCED HTML RENDERING)
    components.html(
        """
        <div style="display:flex; gap:30px; margin-top:35px;">

            <div style="
                flex:1;
                background:linear-gradient(135deg,#1c1c1c,#262626);
                border-left:6px solid #ff9f1a;
                padding:30px;
                border-radius:16px;
                box-shadow:0 10px 25px rgba(0,0,0,0.6);
                transition:all 0.35s ease;
            "
            onmouseover="this.style.transform='translateY(-12px) scale(1.03)';
                         this.style.boxShadow='0 25px 45px rgba(255,159,26,0.55)'"
            onmouseout="this.style.transform='none';
                        this.style.boxShadow='0 10px 25px rgba(0,0,0,0.6)'"
            >
                <div style="font-size:26px;font-weight:800;color:#ff9f1a;margin-bottom:14px;">
                    🎯 Mission
                </div>
                <div style="font-size:16px;line-height:1.6;color:#f5f5f5;">
                    To analyze student study habits using Artificial Intelligence
                    and provide actionable insights that improve academic performance.
                </div>
            </div>

            <div style="
                flex:1;
                background:linear-gradient(135deg,#1c1c1c,#262626);
                border-left:6px solid #ff9f1a;
                padding:30px;
                border-radius:16px;
                box-shadow:0 10px 25px rgba(0,0,0,0.6);
                transition:all 0.35s ease;
            "
            onmouseover="this.style.transform='translateY(-12px) scale(1.03)';
                         this.style.boxShadow='0 25px 45px rgba(255,159,26,0.55)'"
            onmouseout="this.style.transform='none';
                        this.style.boxShadow='0 10px 25px rgba(0,0,0,0.6)'"
            >
                <div style="font-size:26px;font-weight:800;color:#ff9f1a;margin-bottom:14px;">
                    🌍 Vision
                </div>
                <div style="font-size:16px;line-height:1.6;color:#f5f5f5;">
                    To build an intelligent, personalized learning support system
                    that helps every student study smarter, not harder.
                </div>
            </div>

        </div>
        """,
        height=280
    )
    st.markdown("""
<h2 style="
    color:#ff9f1a;
    margin-bottom:18px;
    font-weight:800;
">
    🚀 Key Features
</h2>

<ul style="
    color:#f5f5f5;
    font-size:16px;
    line-height:1.9;
    padding-left:22px;
">
    <li>AI-based analysis of student study habits including study, sleep, and play time.</li>
    <li>Machine Learning model for predicting student academic performance.</li>
    <li>Support for real-world datasets using CSV and Excel file uploads.</li>
    <li>Single student prediction and bulk prediction for multiple students.</li>
    <li>Personalized study habit recommendations based on predicted performance.</li>
    <li>Interactive dashboard designed for academic evaluation and internships.</li>
</ul>
""", unsafe_allow_html=True)

    

       

# MODEL TRAINING
# -------------------------------------------------
def model_training():
    
    page_header("MODEL TRAINING")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Dataset (CSV / Excel)", ["csv", "xlsx"])

    if file:
        # Load dataset
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
        st.dataframe(df.head())

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            st.error("❌ Dataset has no numeric columns for training")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.info("Only numeric columns are used for Machine Learning")

        # Select target column (Test Score / Marks / etc.)
        target = st.selectbox(
            "Select Target Column (Test Score / Marks)",
            numeric_df.columns
        )

        # Select feature columns (exclude target)
        features = st.multiselect(
            "Select Feature Columns (Study, Sleep, Play, Other)",
            [c for c in numeric_df.columns if c != target]
        )

        # Train button
        if st.button("Train Model"):
            if not features:
                st.warning("⚠️ Please select at least one feature column")
            else:
                X = numeric_df[features]
                y = numeric_df[target] / 10   # normalize to 0–1


                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                # ✅ STORE EVERYTHING FOR FUTURE PREDICTIONS
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.target_name = target   # ⭐ IMPORTANT

                st.success("✅ Model trained successfully")

                # Optional metrics (good for viva)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                st.info(f"📊 Training R² Score: {round(train_score, 3)}")
                st.info(f"📊 Testing R² Score: {round(test_score, 3)}")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
def single_prediction():
    page_header("SINGLE PREDICTION")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Safety check
    if "model" not in st.session_state or "features" not in st.session_state:
        st.warning("⚠️ Please train the model first")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.info("Enter values to predict Test Score (Range: 1–10)")

    values = []

    # Collect inputs
    for feature in st.session_state.features:
        val = st.number_input(
            f"{feature}",
            min_value=0.0,
            step=0.5,
            key=f"single_{feature}"
        )
        values.append(val)

    if st.button("Predict"):
        # Block empty input
        if all(v == 0 for v in values):
            st.warning("⚠️ Please enter valid study habit values before prediction.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # ✅ Predict directly (NO scaling)
        prediction = st.session_state.model.predict([values])[0]

        # ✅ Clamp strictly to 1–10
        prediction = round(max(1, min(10, prediction)), 2)

        st.success(f"🎯 Predicted Test Score: {prediction}")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------
# HELPER: CLEAN INPUT DATA (CSV + EXCEL SAFE)
# -------------------------------------------------
def clean_input_data(df, features):
    X = df[features]

    # Convert everything to numeric (text → NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop fully empty rows (CSV junk rows)
    X = X.dropna(how="all")

    # Ensure all required feature columns exist
    X = X.reindex(columns=features, fill_value=0)

    # FINAL GUARANTEE: no NaN at all
    X = X.fillna(0)

    return X
#--------------------------------------------------
# BATCH PREDICTION WITH INSIGHTS & COMPARISON 
# -------------------------------------------------
def batch_prediction():
    page_header("BATCH PREDICTION & DATA INSIGHTS")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # -----------------------------
    # SAFETY CHECK – MODEL TRAINED?
    # -----------------------------
    if (
        "model" not in st.session_state or
        "features" not in st.session_state or
        "target_name" not in st.session_state
    ):
        st.warning("⚠️ Please train the model first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    model = st.session_state["model"]
    features = st.session_state["features"]

    # -----------------------------
    # FILE UPLOAD
    # -----------------------------
    file = st.file_uploader(
        "Upload Dataset WITHOUT Test Score (CSV / Excel)",
        type=["csv", "xlsx"]
    )

    if not file:
        st.info("📂 Upload a dataset to perform batch prediction.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # VALIDATE REQUIRED COLUMNS
    # -----------------------------
    missing_cols = [col for col in features if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # PREDICTION
    # -----------------------------
    X = df[features]
    pred_col = f"Predicted {st.session_state.target_name}"
    pred_col = "Predicted Test Score"
    df[pred_col] = df.apply(calculate_test_score, axis=1)


    # -----------------------------
    # PERFORMANCE LABEL
    # -----------------------------
    pred_col = f"Predicted {st.session_state.target_name}"

    df["Performance"] = df[pred_col].apply(
    lambda x: "Top Performer" if x >= 7 else "Low Performer"
)


    st.subheader("✅ Batch Prediction Results")
    st.dataframe(df)

    # -----------------------------
    # BAR CHART – PREDICTED SCORES
    # -----------------------------
    pred_col = f"Predicted {st.session_state.target_name}"

    chart_df = df[[pred_col]].copy()
    chart_df[pred_col] = chart_df[pred_col] + np.random.normal(0, 0.001, len(chart_df))
    chart_df.index = range(1, len(chart_df) + 1)

    st.bar_chart(chart_df)

    # -----------------------------
    # TOP vs LOW COMPARISON
    # -----------------------------
    st.subheader("⚖️ Top vs Low Performer Comparison")
    top_df = df[df["Performance"] == "Top Performer"]
    low_df = df[df["Performance"] == "Low Performer"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Top Performers (Score ≥ 7)")
        st.metric("Count", len(top_df))
        if not top_df.empty:
            st.dataframe(top_df)

    with col2:
        st.markdown("### 🔴 Low Performers (Score < 6)")
        st.metric("Count", len(low_df))
        if not low_df.empty:
            st.dataframe(low_df)

    # -----------------------------
    # DOWNLOAD RESULT
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction Results (CSV)",
        data=csv,
        file_name="batch_prediction_results.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------
# STUDENT RECOMMENDATION
# -------------------------------------------------

def student_recommendation(row):
    """
    Performance & Suggestion logic based on Predicted Score
    High    : >= 7
    Average : 6 – 6.9
    Low     : < 6
    """
    score = row[f"Predicted {st.session_state.target_name}"]

    if score >= 7:
        return pd.Series([
            "High Performance",
            "Good work! Maintain your current study routine.",
            "green"
        ])
    elif score >= 6:
        return pd.Series([
            "Average Performance",
            "Increase focused study hours and reduce distractions.",
            "yellow"
        ])
    else:
        return pd.Series([
            "Low Performance",
            "Needs improvement. Revise study plan and seek guidance.",
            "red"
        ])


def recommendation():

    page_header("STUDENT RECOMMENDATION")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # -----------------------------
    # SAFETY CHECK
    # -----------------------------
    if (
        "model" not in st.session_state or
        "features" not in st.session_state or
        "target_name" not in st.session_state
    ):
        st.warning("⚠️ Please train the model first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # FILE UPLOAD
    # -----------------------------
    file = st.file_uploader(
        "Upload Student Dataset (WITHOUT Test Score)",
        ["csv", "xlsx"]
    )

    if not file:
        st.info("📂 Upload a dataset to generate recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # CHECK REQUIRED FEATURES
    # -----------------------------
    missing_cols = [
        col for col in st.session_state.features
        if col not in df.columns
    ]

    if missing_cols:
        st.error(f"❌ Missing required feature columns: {missing_cols}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # PREDICTION
    # -----------------------------
    X_new = df[st.session_state.features]
    pred_col = f"Predicted {st.session_state.target_name}"

    pred_col = "Predicted Test Score"
    df[pred_col] = df.apply(calculate_test_score, axis=1)


    # -----------------------------
    # GENERATE RECOMMENDATIONS
    # -----------------------------
    df[["Performance", "Suggestion", "Status"]] = df.apply(
        student_recommendation,
        axis=1
    )

    # -----------------------------
    # DISPLAY REPORT
    # -----------------------------
    st.subheader("📋 Student Recommendation Report")

    def color_status(val):
        if val == "green":
            return "background-color:#2ecc71;color:black;font-weight:bold"
        elif val == "yellow":
            return "background-color:#f1c40f;color:black;font-weight:bold"
        else:
            return "background-color:#e74c3c;color:white;font-weight:bold"

    st.dataframe(
        df.style.applymap(color_status, subset=["Status"])
    )

    # -----------------------------
    # SUMMARY
    # -----------------------------
    st.subheader("📊 Recommendation Summary")
    summary = df["Performance"].value_counts()
    st.write(summary)

    # -----------------------------
    # DOWNLOAD CSV
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Recommendation Report (CSV)",
        data=csv,
        file_name="student_recommendations.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🤖 Model Training", "🔍 Single Prediction", "📂 Batch Prediction", "📘 Recommendation"]
)


    if "Home" in page:
        home()
    elif "Model Training" in page:
        model_training()
    elif "Single Prediction" in page:
        single_prediction()
    elif "Batch Prediction" in page:
        batch_prediction()
    elif "Recommendation" in page:
        recommendation()

